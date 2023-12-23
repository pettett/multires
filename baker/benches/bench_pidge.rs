use baker::pidge::Pidge;
use criterion::{criterion_group, criterion_main, Criterion};

fn use_pidge() {
    let mut pidge = Pidge::with_capacity(10);

    for i in 0..10usize {
        pidge.insert(i, i);
    }

    pidge.wipe(1);
    pidge.wipe(2);
    pidge.wipe(3);
    pidge.wipe(9);


    let mut t = 0;

    for _p in pidge.iter() {
        t += 1;
    }

	assert_eq!(t, pidge.len());
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("pidge 10", |b| b.iter(|| use_pidge()));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
