#![cfg_attr(feature = "nightly", feature(integer_atomics))]

use std;

use atomics::*;
use super::*;
use std::sync::atomic::{AtomicUsize, AtomicPtr, Ordering};
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
    eval: Spec::Eval,
    manager: Spec,
    num_nodes: AtomicUsize,
}

pub struct MoveInfo<Spec: MCTS> {
    mov: Move<Spec>,
    move_evaluation: MoveEvaluation<Spec>,
    child: AtomicPtr<SearchNode<Spec>>,
    visits: AtomicUsize,
    sum_evaluations: AtomicI64,
}

pub struct SearchNode<Spec: MCTS> {
    moves: Vec<MoveInfo<Spec>>,
    data: Spec::NodeData,
    evaln: StateEvaluation<Spec>,
}

impl<Spec: MCTS> SearchNode<Spec> {
    fn new(moves: Vec<MoveInfo<Spec>>,
            evaln: StateEvaluation<Spec>) -> Self {
        Self {
            moves,
            data: Default::default(),
            evaln,
        }
    }
}

impl<Spec: MCTS> MoveInfo<Spec> {
    fn new(mov: Move<Spec>, move_evaluation: MoveEvaluation<Spec>) -> Self {
        MoveInfo {
            mov,
            move_evaluation,
            child: AtomicPtr::default(),
            visits: AtomicUsize::new(0),
            sum_evaluations: AtomicI64::new(0),
        }
    }

    pub fn get_move(&self) -> &Move<Spec> {
        &self.mov
    }

    pub fn move_evaluation(&self) -> &MoveEvaluation<Spec> {
        &self.move_evaluation
    }

    pub fn visits(&self) -> u64 {
        self.visits.load(Ordering::Relaxed) as u64
    }

    pub fn sum_rewards(&self) -> i64 {
        self.sum_evaluations.load(Ordering::Relaxed) as i64
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
        if self.visits() == 0 {
            write!(f, "{} [0 visits]", self.mov)
        } else {
            write!(f, "{} [{} visit{}] [{} avg reward]",
                self.mov, self.visits(), if self.visits() == 1 {""} else {"s"},
                self.sum_rewards() as f64 / self.visits() as f64)
        }
    }
}

impl<Spec: MCTS> Debug for MoveInfo<Spec> where Move<Spec>: Debug {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        if self.visits() == 0 {
            write!(f, "{:?} [0 visits]", self.mov)
        } else {
            write!(f, "{:?} [{} visit{}] [{} avg reward]",
                self.mov, self.visits(), if self.visits() == 1 {""} else {"s"},
                self.sum_rewards() as f64 / self.visits() as f64)
        }
    }
}

impl<Spec: MCTS> Drop for MoveInfo<Spec> {
    fn drop(&mut self) {
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

impl<Spec: MCTS> SearchTree<Spec> {
    pub fn new(state: Spec::State, manager: Spec, tree_policy: Spec::TreePolicy, eval: Spec::Eval)
            -> Self {
        let root_node = create_node(&eval, &tree_policy, &state, None);
        Self {
            root_state: state,
            root_node,
            manager,
            tree_policy,
            eval,
            num_nodes: AtomicUsize::new(1),
        }
    }

    pub fn reset(self) -> Self {
        Self::new(self.root_state, self.manager, self.tree_policy, self.eval)
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
        let total_nodes = self.num_nodes.fetch_add(1, Ordering::Relaxed);
        if total_nodes >= self.manager.node_limit() {
            self.num_nodes.fetch_sub(1, Ordering::Relaxed);
            return false;
        }
        let mut state = self.root_state.clone();
        let mut path: SmallVec<[&MoveInfo<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut players: SmallVec<[Player<Spec>; LARGE_DEPTH]> = SmallVec::new();
        let mut did_we_create = false;
        let mut node = &self.root_node;
        loop {
            if node.moves.len() == 0 {
                break;
            }
            let choice = self.tree_policy.choose_child(node.moves.iter(), self.make_handle(node, tld));
            let child_visits = choice.visits.fetch_add(1, Ordering::Relaxed) + 1;
            choice.sum_evaluations.fetch_sub(self.manager.virtual_loss() as isize, Ordering::Relaxed);
            players.push(state.current_player());
            path.push(choice);
            state.make_move(&choice.mov);
            let mut child;
            loop {
                child = choice.child.load(Ordering::Acquire) as *const SearchNode<Spec>;
                did_we_create = false;
                if child == null() {
                    let new_child = create_node(&self.eval, &self.tree_policy, &state,
                        Some(self.make_handle(node, tld)));
                    let new_child = Box::into_raw(Box::new(new_child)); // move to heap
                    let result = choice.child.compare_and_swap(null_mut(), new_child, Ordering::Release);
                    if result == null_mut() {
                        // compare and swap was successful
                        did_we_create = true;
                        child = new_child;
                        self.num_nodes.fetch_add(1, Ordering::Relaxed);
                        break;
                    } else {
                        // self.contention_events.fetch_add(1, Ordering::Relaxed);
                        // someone else expanded this child before we did
                        unsafe {
                            Box::from_raw(new_child);
                        }
                    }
                } else {
                    break;
                }
            }
            assert!(child != null());
            node = unsafe {
                &*child
            };
            if child_visits as u64 <= self.manager.visits_before_expansion() {
                break;
            }
        }
        self.num_nodes.fetch_sub(1, Ordering::Relaxed);
        self.finish_playout(did_we_create, &state, &path, &players, tld, node);
        true
    }

    fn finish_playout(&self, did_we_create: bool, state: &Spec::State,
            path: &[&MoveInfo<Spec>], players: &[Player<Spec>], tld: &mut ThreadData<Spec>,
            final_node: &SearchNode<Spec>) {
        let new_evaln = if did_we_create {
            None
        } else {
            Some(self.eval.evaluate_existing_state(state, &final_node.evaln, self.make_handle(final_node, tld)))
        };
        let evaln = new_evaln.as_ref().unwrap_or(&final_node.evaln);
        // Last ones more likely to be in cache
        for (move_info, player) in path.iter().zip(players.iter()).rev() {
            let delta =
                  self.eval.interpret_evaluation_for_player(evaln, player)
                + self.manager.virtual_loss();
            move_info.sum_evaluations.fetch_add(delta as isize, Ordering::Relaxed);
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

    pub fn principal_variation(&self, num_moves: usize) -> Vec<Move<Spec>> {
        let mut result = Vec::new();
        let mut crnt = &self.root_node;
        while crnt.moves.len() != 0 && result.len() < num_moves {
            let choice = self.manager.select_child_after_search(&crnt.moves);
            result.push(choice.mov.clone());
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
}

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
