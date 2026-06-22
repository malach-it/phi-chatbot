pub(crate) fn add_curve_points(total: &mut Option<Vec<f64>>, points: &[f64]) {
    let total = total.get_or_insert_with(|| vec![0.0; points.len()]);

    if total.len() < points.len() {
        total.resize(points.len(), 0.0);
    }

    for (index, point) in points.iter().enumerate() {
        total[index] += point;
    }
}

pub(crate) fn draw_curve(points: &[f64], width: usize) -> String {
    if points.is_empty() || width == 0 {
        return String::new();
    }

    let min = points.iter().copied().fold(f64::INFINITY, f64::min);
    let max = points.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let scale = (max - min).max(1e-9);
    let width = width.max(2);
    let height = 10;
    let mut sampled_rows = Vec::with_capacity(width);

    for column in 0..width {
        let x = column as f64 / (width - 1) as f64;
        let value = interpolate_points(points, x);
        let normalized = ((value - min) / scale).clamp(0.0, 1.0);
        let row = ((1.0 - normalized) * (height - 1) as f64).round() as usize;
        sampled_rows.push(row);
    }

    let mut output = String::new();

    for row in 0..height {
        let y = max - (row as f64 / (height - 1) as f64) * scale;
        output.push_str(&format!("    {y:>8.4} |"));

        for sampled_row in &sampled_rows {
            output.push(if *sampled_row == row { '*' } else { ' ' });
        }

        output.push('\n');
    }

    output.push_str(&format!("             +{}\n", "-".repeat(width)));
    output.push_str("              0.00");
    if width > 8 {
        output.push_str(&" ".repeat(width.saturating_sub(8)));
    }
    output.push_str("1.00\n");

    for (index, value) in points.iter().enumerate() {
        let x = index as f64 / (points.len() - 1).max(1) as f64;
        output.push_str(&format!("    point x={x:.2} y={value:>8.4}\n"));
    }

    output
}

fn interpolate_points(points: &[f64], x: f64) -> f64 {
    if points.len() == 1 {
        return points[0];
    }

    let scaled = x.clamp(0.0, 1.0) * (points.len() - 1) as f64;
    let lower_index = scaled.floor() as usize;
    let upper_index = (lower_index + 1).min(points.len() - 1);
    let upper_weight = scaled - lower_index as f64;
    let lower_weight = 1.0 - upper_weight;

    points[lower_index] * lower_weight + points[upper_index] * upper_weight
}
