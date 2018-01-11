extern crate mcts;

use mcts::*;
use mcts::tree_policy::*;

#[derive(Clone, Copy)]
struct CountingGame(i64);

impl GameState for CountingGame {
    type Move = CountingGame;
    type Player = ();

    fn current_player(&self) -> Self::Player {
        ()
    }

    fn available_moves(&self) -> Vec<Self::Move> {
        let x = self.0;
        vec![CountingGame(x - 1), CountingGame(x + 1)]
    }

    fn make_move(&mut self, mov: &Self::Move) {
        *self = *mov;
    }
}

struct MyEvaluator {}

impl Evaluator<MyMCTS> for MyEvaluator {
    type StateEvaluation = i64;

    fn evaluate_new_state(&self, state: &CountingGame, moves: &[CountingGame],
        _: SearchHandle<MyMCTS>)
        -> (Vec<f64>, i64) {
        (moves.iter().map(|_| 0.0).collect(), state.0)
    }

    fn interpret_evaluation_for_player(&self, evaln: &i64, _player: &()) -> i64 {
        *evaln
    }

    fn evaluate_existing_state(&self, _: &CountingGame,  evaln: &i64, _: SearchHandle<MyMCTS>) -> i64 {
        *evaln
    }
}

struct MyMCTS {}

impl MCTS for MyMCTS {
    type State = CountingGame;
    type Eval = MyEvaluator;
    type NodeData = ();
    type ThreadLocalData = PolicyRng;
    type GlobalData = ();
    type TreePolicy = UCTPolicy;
}

fn main() {
    let game = CountingGame(0);
}
