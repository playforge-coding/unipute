use unipute::kernel;

#[kernel(workgroup_size(64))]
fn wrong_type(values: &[String]) {
    let first = values[0];
}

fn main() {}
