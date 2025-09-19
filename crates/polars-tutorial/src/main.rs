use polars::prelude::*;

fn main() -> Result<(), PolarsError> {
    demo_lazy()?;
    demo_aggregation()?;
    demo_with_columns()?;
    demo_groupby_and_window()?;
    Ok(())
}

fn demo_lazy() -> Result<(), PolarsError> {
    let df = df![
        "category" => &["A", "A", "B", "C", "B", "A"],
        "values" => &[10, 20, 5, 40, 50, 60],
        "counts" => &[1, 2, 3, 1, 5, 3]
    ]?;

    let lazy_plan = df
        .lazy()
        .filter(col("category").eq(lit("A")))
        .filter(col("counts").gt(lit(1)))
        .select([col("values").sum()]);

    let result_lazy = lazy_plan.collect()?;
    let sum_lazy = result_lazy.column("values")?.get(0)?.try_extract::<i32>()?;
    println!(
        "[Lazy 模式] 'A' 類別且 'counts' > 1 的 'values' 總和: {:?}",
        sum_lazy
    );

    Ok(())
}

fn demo_aggregation() -> Result<(), PolarsError> {
    let df = df![
        "category" => &["A", "A", "B", "B", "C", "C"],
        "value" => &[10, 20, 30, 40, 50, 60]
    ]?;

    let result = df
        .lazy()
        .group_by(["category"])
        .agg([
            col("value").sum().alias("total"),
            col("value").mean().alias("average"),
            col("value").count().alias("count"),
        ])
        .sort(["category"], Default::default())
        .collect()?;

    println!("\n[Aggregation] 分組統計結果:\n{}", result);
    Ok(())
}

fn demo_with_columns() -> Result<(), PolarsError> {
    println!("\n--- 範例 3: with_columns 上下文操作 ---");

    let df = df![
        "A" => &[1.0, 2.0, 3.0, 4.0, 5.0],
        "B" => &[5.0, 4.0, 3.0, 2.0, 1.0],
        "text" => &["hello", "world", "polars", "is", "fast"]
    ]?;

    let df_transformed = df
        .lazy()
        .with_columns(&[
            (col("A") + col("B")).alias("A_plus_B"),
            when(col("A").gt(lit(3.0)))
                .then(lit("high"))
                .otherwise(lit("low"))
                .alias("A_category"),
        ])
        .collect()?;

    println!("\n[with_columns] 一次性新增多個欄位:\n{}", df_transformed);
    Ok(())
}

fn demo_groupby_and_window() -> Result<(), PolarsError> {
    println!("\n--- 範例 4: GroupBy 與 Window Functions ---");

    let df = df![
        "department" => &["HR", "IT", "IT", "HR", "Sales", "Sales", "IT"],
        "employee" => &["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"],
        "salary" => &[60000, 80000, 95000, 75000, 120000, 110000, 150000]
    ]?;

    let aggregated_df = df
        .clone()
        .lazy()
        .group_by(["department"])
        .agg(&[
            col("salary").sum().alias("total_salary"),
            col("salary").mean().alias("average_salary"),
            col("employee").count().alias("employee_count"),
        ])
        .sort(["department"], Default::default())
        .collect()?;
    println!("\n[GroupBy] 按部門聚合計算:\n{}", aggregated_df);

    let window_df = df
        .lazy()
        .with_columns(&[
            (col("salary") * lit(100.0) / col("salary").sum().over(["department"]))
                .alias("salary_%_of_department"),
            (col("salary") / col("salary").max().over(["department"]))
                .alias("salary_ratio_in_department"),
        ])
        .sort(["department", "salary"], Default::default())
        .collect()?;

    println!(
        "\n[Window Function] 在部門內計算薪水佔比與排名:\n{}",
        window_df
    );

    Ok(())
}
