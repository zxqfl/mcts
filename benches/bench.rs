#![feature(test)]

include!("../examples/counting_game.rs");

extern crate test;
use self::test::Bencher;

#[bench]
fn counting_game(b: &mut Bencher) {
    b.iter(|| {
        let mut mcts = MCTSManager::new(CountingGame(0), MyMCTS, UCTPolicy::new(1e6), MyEvaluator);
        mcts.playout_n(100000);
    });
}
