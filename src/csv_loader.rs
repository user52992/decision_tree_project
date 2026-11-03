use csv::Reader;
use std::error::Error;
use std::path::PathBuf;

fn get_path(file_name: &str) -> PathBuf {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("data")
        .join("processed")
        .join(format!("{}.csv", file_name));

    path
}

pub fn read_features(file_name: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut rdr = Reader::from_path(get_path(file_name))?;
    let mut data = Vec::new();

    for res in rdr.records() {
        let record = res?;

        let row: Vec<f64> = record
            .iter()
            .map(|value| value.parse::<f64>())
            .collect::<Result<_, _>>()?;

        data.push(row);
    }

    Ok(data)
}

pub fn read_labels(file_name: &str) -> Result<Vec<usize>, Box<dyn Error>> {
    let mut rdr = Reader::from_path(get_path(file_name))?;
    let mut data = Vec::new();

    for res in rdr.records() {
        let record = res?;

        let label: usize = record[0].parse()?;
        data.push(label);
    }

    Ok(data)
}
