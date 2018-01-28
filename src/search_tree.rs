#![cfg_attr(feature = "nightly", feature(integer_atomics))]

use std;

use atomics::*;
use super::*;
use std::sync::Mutex;
use std::ptr::{null, null_mut};
use smallvec::SmallVec;
use std::fmt;
use std::fmt::{Debug, Display, Formatter};

use tree_policy::TreePolicy;

/// You're not intended to use this class (use an `MCTSManager` instead),
/// but you can use it if you want to manage the threads yourself.
pub struct SearchTree<Spec: MCTS> {
    root_node: SearchNode<Spec>,
    root_state: Spec::State,
    tree_policy: Spec::TreePolicy,
    table: Spec::TranspositionTable,
    eval: Spec::Eval,
    manager: Spec,

    num_nodes: AtomicUsize,
    orphaned: Mutex<Vec<Box<SearchNode<Spec>>>>,
    transposition_table_hits: AtomicUsize,
    delayed_transposition_table_hits: AtomicUsize,
    expansion_contention_events: AtomicUsize,
}

struct NodeStats {
    visits: AtomicUsize,
    sum_evaluations: AtomicI64,
}

pub struct MoveInfo<Spec: MCTS> {
    mov: Move<Spec>,
    move_evaluation: MoveEvaluation<Spec>,
    child: AtomicPtr<SearchNode<Spec>>,
    owned: AtomicBool,
    stats: NodeStats,
}

pub struct SearchNode<Spec: MCTS> {
    moves: Vec<MoveInfo<Spec>>,
    data: Spec::NodeData,
    evaln: StateEvaluation<Spec>,
    stats: NodeStats,
}

impl<Spec: MCTS> SearchNode<Spec> {
    fn new(moves: Vec<MoveInfo<Spec>>,
            evaln: StateEvaluation<Spec>) -> Self {
        Self {
            moves,
            data: Default::default(),
            evaln,
            stats: NodeStats::new(),
        }
    }
}

impl<Spec: MCTS> MoveInfo<Spec> {
    fn new(mov: Move<Spec>, move_evaluation: MoveEvaluation<Spec>) -> Self {
        MoveInfo {
            mov,
            move_evaluation,
            child: AtomicPtr::default(),
            stats: NodeStats::new(),
            owned: AtomicBool::new(false),
        }
    }

    pub fn get_move(&self) -> &Move<Spec> {
        &self.mov
    }

    pub fn move_evaluation(&self) -> &MoveEvaluation<Spec> {
        &self.move_evaluation
    }

    pub fn visits(&self) -> u64 {
        self.stats.visits.load(Ordering::Relaxed) as u64
    }

    pub fn sum_rewards(&self) -> i64 {
        self.stats.sum_evaluations.load(Ordering::Relaxed) as i64
    }

    pub fn child(&self) -> Option<NodeHandle<Spec>> {
        let ptr = self.child.load(Ordering::Relaxed);
        if ptr == null_mut() {
            None
        } else {
            unsafe {Some(NodeHandle {node: &*ptr})}
        }
    }
}

impl<Spec: MCTS> Display for MoveInfo<Spec> where Move<Spec>: Display {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let own_str = if self.owned.load(Ordering::Relaxed) {
            ""
        } else {
            " [child pointer is alias]"
        };
        if self.visits() == 0 {
            write!(f, "{} [0 visits]{}",
                self.mov,
                own_str)
        } else {
            write!(f, "{} [{} visit{}] [{} avg reward]{}",
                self.mov, self.visits(), if self.visits() == 1 {""} else {"s"},
                self.sum_rewards() as f64 / self.visits() as f64,
                own_str)
        }
    }
}

impl<Spec: MCTS> Debug for MoveInfo<Spec> where Move<Spec>: Debug {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        let own_str = if self.owned.load(Ordering::Relaxed) {
            ""
        } else {
            " [child pointer is alias]"
        };
        if self.visits() == 0 {
            write!(f, "{:?} [0 visits]{}",
                self.mov,
                own_str)
        } else {
            write!(f, "{:?} [{} visit{}] [{} avg reward]{}",
                self.mov, self.visits(), if self.visits() == 1 {""} else {"s"},
                self.sum_rewards() as f64 / self.visits() as f64,
                own_str)
        }
    }
}

impl<Spec: MCTS> Drop for MoveInfo<Spec> {
    fn drop(&mut self) {
        if !self.owned.load(Ordering::SeqCst) {
            return;
        }
        let ptr = self.child.load(Ordering::SeqCst);
        if ptr != null_mut() {
            unsafe {
                Box::from_raw(ptr);
            }
        }
    }
}

fn create_node<Spec: MCTS>(eval: &Spec::Eval, policy: &Spec::TreePolicy, state: &Spec::State,
        handle: Option<SearchHandle<Spec>>) -> SearchNode<Spec> {
    let moves = state.available_moves();
    let (move_eval, state_eval) = eval.evaluate_new_state(&state, &moves, handle);
    policy.validate_evaluations(&move_eval);
    let moves = moves.into_iter()
        .zip(move_eval.into_iter())
        .map(|(m, e)| MoveInfo::new(m, e))
        .collect();
    SearchNode::new(moves, state_eval)
}

fn is_cycle<T>(past: &[&T], current: &T) -> bool {
    past.iter().any(|x| *x as *const T == current as *const T)
}

impl<Spec: MCTS> SearchTree<Spec> {
    pub fn new(state: Spec::State, manager: Spec, tree_policy: Spec::TreePolicy, eval: Spec::Eval,
            table: Spec::TranspositionTable) -> Self {
        let root_node = create_node(&eval, &tree_policy, &state, None);
        Self {
            root_state: state,
            root_node,
            manager,
            tree_policy,
            eval,
            table,
            num_nodes: 1.into(),
            orphaned: Mutex::new(Vec::new()),
            transposition_table_hits: 0.into(),
            delayed_transposition_table_hits: 0.into(),
            expansion_contention_events: 0.into(),
        }
    }

    pub fn reset(self) -> Self {
        Self::new(self.root_state, self.manager, self.tree_policy, self.eval, self.table)
    }

    pub fn spec(&self) -> &Spec {
        &self.manager
    }

    pub fn node_count(&self) -> usize {
        self.num_nodes.load(Ordering::SeqCst)
    }

    #[inline(never)]
    pub fn playout(&self, tld: &mut ThreadData<Spec>) -> bool {
        const LARGE_DEPTH: usize = 64;
        let sentinel = IncreaseSentinel::new(&self.num_nodes);
        if sentinel.num_nodes >= self.manager.node_limit() {
            return false;
        }
        let mut state = self.root_state.clone();
        let mut path: SmallVec<[&MoveInfo<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut node_path: SmallVec<[&SearchNode<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut players: SmallVec<[Player<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut did_we_create = false;
        let mut node = &self.root_node;
        loop {
            if node.moves.len() == 0 {
                break;
            }
            if path.len() >= self.manager.max_playout_length() {
                break;
            }
            let choice = self.tree_policy.choose_child(node.moves.iter(), self.make_handle(node, tld));
            choice.stats.down(&self.manager);
            players.push(state.current_player());
            path.push(choice);
            assert!(path.len() <= self.manager.max_playout_length(),
                "playout length exceeded maximum of {} (maybe the transposition table is creating an infinite loop?)",
                self.manager.max_playout_length());
            state.make_move(&choice.mov);
            let (new_node, new_did_we_create) = self.descend(&state, choice, node, tld);
            node = new_node;
            did_we_create = new_did_we_create;
            match self.manager.cycle_behaviour() {
                CycleBehaviour::Ignore => (),
                CycleBehaviour::PanicWhenCycleDetected => if is_cycle(&node_path, node) {
                    panic!("cycle detected! you should do one of the following:\n- make states acyclic\n- remove transposition table\n- change cycle_behaviour()");
                },
                CycleBehaviour::UseCurrentEvalWhenCycleDetected => if is_cycle(&node_path, node) {
                    break;
                },
                CycleBehaviour::UseThisEvalWhenCycleDetected(e) => if is_cycle(&node_path, node) {
                    self.finish_playout(&path, &node_path, &players, tld, &e);
                    return true;
                },
            };
            node_path.push(node);
            node.stats.down(&self.manager);
            if node.stats.visits.load(Ordering::Relaxed) as u64
                    <= self.manager.visits_before_expansion() {
                break;
            }
        }
        let new_evaln = if did_we_create {
            None
        } else {
            Some(self.eval.evaluate_existing_state(&state, &node.evaln, self.make_handle(node, tld)))
        };
        let evaln = new_evaln.as_ref().unwrap_or(&node.evaln);
        self.finish_playout(&path, &node_path, &players, tld, evaln);
        true
    }

    fn descend<'a, 'b>(&'a self, state: &Spec::State, choice: &MoveInfo<Spec>,
            current_node: &'b SearchNode<Spec>, tld: &'b mut ThreadData<Spec>)
            -> (&'a SearchNode<Spec>, bool) {
        let child = choice.child.load(Ordering::Relaxed) as *const _;
        if child != null() {
            return unsafe { (&*child, false) };
        }
        if let Some(node) = self.table.lookup(state, self.make_handle(current_node, tld)) {
            let child = choice.child.compare_and_swap(
                null_mut(),
                node as *const _ as *mut _,
                Ordering::Relaxed) as *const _;
            if child == null() {
                self.transposition_table_hits.fetch_add(1, Ordering::Relaxed);
                return (node, false);
            } else {
                return unsafe { (&*child, false) };
            }
        }
        let created = create_node(&self.eval, &self.tree_policy, state, Some(self.make_handle(current_node, tld)));
        let created = Box::into_raw(Box::new(created));
        let other_child = choice.child.compare_and_swap(
            null_mut(),
            created,
            Ordering::Relaxed);
        if other_child != null_mut() {
            self.expansion_contention_events.fetch_add(1, Ordering::Relaxed);
            unsafe {
                Box::from_raw(created);
                return (&*other_child, false);
            }
        }
        if let Some(existing) = self.table.insert(state, unsafe {&*created}, self.make_handle(current_node, tld)) {
            self.delayed_transposition_table_hits.fetch_add(1, Ordering::Relaxed);
            let existing_ptr = existing as *const _ as *mut _;
            choice.child.store(existing_ptr, Ordering::Relaxed);
            self.orphaned.lock().unwrap().push(unsafe { Box::from_raw(created) });
            return (existing, false);
        }
        choice.owned.store(true, Ordering::Relaxed);
        self.num_nodes.fetch_add(1, Ordering::Relaxed);
        unsafe { (&*created, true) }
    }

    fn finish_playout(&self,
            path: &[&MoveInfo<Spec>],
            node_path: &[&SearchNode<Spec>],
            players: &[Player<Spec>],
            tld: &mut ThreadData<Spec>,
            evaln: &StateEvaluation<Spec>) {
        for ((move_info, player), node) in
                path.iter()
                .zip(players.iter())
                .zip(node_path.iter())
                .rev() {
            let evaln_value = self.eval.interpret_evaluation_for_player(evaln, player);
            node.stats.up(&self.manager, evaln_value);
            move_info.stats.replace(&node.stats);
            unsafe {
                self.manager.on_backpropagation(&evaln, self.make_handle(&*move_info.child.load(Ordering::Relaxed), tld));
            }
        }
        self.manager.on_backpropagation(&evaln, self.make_handle(&self.root_node, tld));
    }

    fn make_handle<'a>(&'a self, node: &'a SearchNode<Spec>, tld: &'a mut ThreadData<Spec>)
            -> SearchHandle<'a, Spec> {
        SearchHandle {node, tld, manager: &self.manager}
    }

    pub fn root_state(&self) -> &Spec::State {
        &self.root_state
    }
    pub fn root_node(&self) -> NodeHandle<Spec> {
        NodeHandle {
            node: &self.root_node
        }
    }

    pub fn principal_variation(&self, num_moves: usize) -> Vec<MoveInfoHandle<Spec>> {
        let mut result = Vec::new();
        let mut crnt = &self.root_node;
        while crnt.moves.len() != 0 && result.len() < num_moves {
            let choice = self.manager.select_child_after_search(&crnt.moves);
            result.push(choice);
            let child = choice.child.load(Ordering::SeqCst) as *const SearchNode<Spec>;
            if child == null() {
                break;
            } else {
                unsafe {
                    crnt = &*child;
                }
            }
        }
        result
    }

    pub fn diagnose(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("{} nodes\n", thousands_separate(self.num_nodes.load(Ordering::Relaxed))));
        s.push_str(&format!("{} transposition table hits\n", thousands_separate(self.transposition_table_hits.load(Ordering::Relaxed))));
        s.push_str(&format!("{} delayed transposition table hits\n", thousands_separate(self.delayed_transposition_table_hits.load(Ordering::Relaxed))));
        s.push_str(&format!("{} expansion contention events\n", thousands_separate(self.expansion_contention_events.load(Ordering::Relaxed))));
        s.push_str(&format!("{} orphaned nodes\n", self.orphaned.lock().unwrap().len()));
        s
    }
}

pub type MoveInfoHandle<'a, Spec> = &'a MoveInfo<Spec>;

impl<Spec: MCTS> SearchTree<Spec> where Move<Spec>: Debug {
    pub fn debug_moves(&self) {
        let mut moves: Vec<&MoveInfo<Spec>> = self.root_node.moves.iter().collect();
        moves.sort_by_key(|x| -(x.visits() as i64));
        for mov in moves {
            println!("{:?}", mov);
        }
    }
}

impl<Spec: MCTS> SearchTree<Spec> where Move<Spec>: Display {
    pub fn display_moves(&self) {
        let mut moves: Vec<&MoveInfo<Spec>> = self.root_node.moves.iter().collect();
        moves.sort_by_key(|x| -(x.visits() as i64));
        for mov in moves {
            println!("{}", mov);
        }
    }
}

#[derive(Clone, Copy)]
pub struct NodeHandle<'a, Spec: 'a + MCTS> {
    node: &'a SearchNode<Spec>,
}

impl<'a, Spec: MCTS> NodeHandle<'a, Spec> {
    pub fn data(&self) -> &'a Spec::NodeData {
        &self.node.data
    }
    pub fn moves(&self) -> Moves<Spec> {
        Moves {
            iter: self.node.moves.iter()
        }
    }
    pub fn into_raw(&self) -> *const () {
        self.node as *const _ as *const ()
    }
    pub unsafe fn from_raw(ptr: *const ()) -> Self {
        NodeHandle {
            node: &*(ptr as *const SearchNode<Spec>)
        }
    }
}

#[derive(Clone)]
pub struct Moves<'a, Spec: 'a + MCTS> {
    iter: std::slice::Iter<'a, MoveInfo<Spec>>,
}

impl<'a, Spec: 'a + MCTS> Iterator for Moves<'a, Spec> {
    type Item = &'a MoveInfo<Spec>;
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

pub struct SearchHandle<'a, Spec: 'a + MCTS> {
    node: &'a SearchNode<Spec>,
    tld: &'a mut ThreadData<Spec>,
    manager: &'a Spec,
}

impl<'a, Spec: MCTS> SearchHandle<'a, Spec> {
    pub fn node(&self) -> NodeHandle<'a, Spec> {
        NodeHandle {node: self.node}
    }
    pub fn thread_local_data(&mut self) -> &mut ThreadData<Spec> {
        self.tld
    }
    pub fn mcts(&self) -> &'a Spec {
        self.manager
    }
}

impl NodeStats {
    fn new() -> Self {
        NodeStats {
            sum_evaluations: AtomicI64::new(0),
            visits: AtomicUsize::new(0),
        }
    }
    fn down<Spec: MCTS>(&self, manager: &Spec) {
        self.sum_evaluations.fetch_sub(manager.virtual_loss() as FakeI64, Ordering::Relaxed);
        self.visits.fetch_add(1, Ordering::Relaxed);
    }
    fn up<Spec: MCTS>(&self, manager: &Spec, evaln: i64) {
        let delta = evaln + manager.virtual_loss();
        self.sum_evaluations.fetch_add(delta as FakeI64, Ordering::Relaxed);
    }
    fn replace(&self, other: &NodeStats) {
        self.visits.store(other.visits.load(Ordering::Relaxed), Ordering::Relaxed);
        self.sum_evaluations.store(other.sum_evaluations.load(Ordering::Relaxed), Ordering::Relaxed);
    }
}

struct IncreaseSentinel<'a> {
    x: &'a AtomicUsize,
    num_nodes: usize
}

impl<'a> IncreaseSentinel<'a> {
    fn new(x: &'a AtomicUsize) -> Self {
        let num_nodes = x.fetch_add(1, Ordering::Relaxed);
        Self {x, num_nodes}
    }
}

impl<'a> Drop for IncreaseSentinel<'a> {
    fn drop(&mut self) {
        self.x.fetch_sub(1, Ordering::Relaxed);
    }
}
