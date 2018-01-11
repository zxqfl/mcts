use super::GameState;
use config::MCTSConfig;

struct SearchTree<State: GameState> {
    root: SearchNode,
    config: MCTSConfig,
}

struct SearchNode {}

impl<State: GameState> SearchTree<State> {
    pub fn playout(&mut self) {
        unimplemented!()
    }
}

impl<State: GameState> SearchTree<State> {
    pub fn playout_until<Predicate: FnMut() -> bool>(&mut self, pred: Predicate) {
        unimplemented!()
    }
    pub fn playout_n(&mut self, n: u64) {
        for _ in 0..n {
            self.playout();
        }
    }
    pub fn playout_parallel_until<Predicate: FnMut() -> bool>
            (&mut self, pred: Predicate, num_threads: usize) {
        unimplemented!()
    }
    pub fn playout_n_parallel(&mut self, n: u64, num_threads: usize) {
        unimplemented!()
    }
}
