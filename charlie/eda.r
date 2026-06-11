library(data.table)
library(collapse)
library(arrow)
library(ggplot2)

# 1. Load Data
dt <- setDT(read_parquet("data/dt_clean2.parquet"))
user_outcomes <- setDT(read_parquet("data/user_outcomes2.parquet"))

# 2. Define flattening function (from 02_sampling_flattening.qmd)
flatten_journey <- function(events_dt) {
	setorder(events_dt, id, event_timestamp)

	events_dt[, ts_int := as.integer(event_timestamp)]
	events_dt[, cutoff_int := as.integer(cutoff_time)]
	events_dt[, gap_s := ts_int - shift(ts_int), by = id]

	summary_dt <- events_dt[,
		{
			first_ts <- event_timestamp[1]
			last_ts <- event_timestamp[.N]
			first_ts_int <- ts_int[1]
			last_ts_int <- ts_int[.N]
			user_cutoff_int <- cutoff_int[1]

			.(
				first_action_ts = first_ts,
				last_action_ts = last_ts,
				journey_length_s = last_ts_int - first_ts_int,
				days_inactive = as.integer((user_cutoff_int - last_ts_int) / 86400),
				total_actions = .N,
				mean_gap_sec = if (.N > 1) mean(gap_s, na.rm = TRUE) else NA_real_,
				median_gap_sec = if (.N > 1) fmedian(gap_s, na.rm = TRUE) else NA_real_,
				max_gap_sec = if (.N > 1) max(gap_s, na.rm = TRUE) else NA_integer_
			)
		},
		by = id
	]

	events_dt[, c("ts_int", "cutoff_int", "gap_s") := NULL]

	counts <- dcast(
		events_dt,
		id ~ event_name,
		value.var = "event_name",
		fun.aggregate = length,
		fill = 0
	)
	setnames(
		counts,
		setdiff(names(counts), "id"),
		paste0("n_", setdiff(names(counts), "id"))
	)

	return(merge(summary_dt, counts, by = "id"))
}

# 3. Flatten the full dataset
# Assuming a 'cutoff_time' for all users as the max timestamp in the data
dt[, cutoff_time := max(event_timestamp)]
flat_dt <- flatten_journey(dt)

# Merge with outcomes
flat_dt <- merge(flat_dt, user_outcomes, by = "id")

# 4. Create EDA Plots

outcome_colors <- c(
  "failure" = "#FF9999",    # Pastel Red
  "incomplete" = "#B3CDE3", # Pastel Blue/Grey
  "success" = "#CCEBC5"     # Pastel Green
)

# Plot 1: Journey Length by Outcome (Histogram)
ggplot(flat_dt, aes(x = journey_length_s / 86400, y = after_stat(density), color = final_outcome)) +
	geom_freqpoly(binwidth=10, linewidth = 1) +
	labs(title = "Distribution of Journey Length by Outcome", x = "Journey Length (days)") +
  scale_x_continuous(limits = c(0, NA)) + 
  scale_color_manual(values = outcome_colors) +
	theme_minimal()

ggsave("charlie/eda-plots/journey_length_distr.png", width = 8, height = 5)

# Plot 2: Total Actions by Outcome
ggplot(flat_dt, aes(x = final_outcome, y = total_actions, fill = final_outcome)) +
	geom_boxplot() +
  scale_fill_manual(values = outcome_colors) + 
	scale_y_log10() +
	labs(title = "Total Actions by Outcome", x = "Outcome", y = "Total Actions (log scale)") +
	theme_minimal()

ggsave("charlie/eda-plots/total_actions_by_outcome.png", width = 8, height = 5)


# 3. Aggregating Top Actions by Outcome
library(tidyr)

# Reshape to long format for event counts
event_cols <- grep("^n_", names(flat_dt), value = TRUE)
long_events <- melt(
  flat_dt, 
  id.vars = "final_outcome", 
  measure.vars = event_cols, 
  variable.name = "event_name", 
  value.name = "count"
)

# Summarize average usage per user by outcome
action_summary <- long_events[, .(avg_count = mean(count)), by = .(final_outcome, event_name)]

# Take the top 10 most common actions overall
top_actions <- action_summary[, .(total = sum(avg_count)), by = event_name][order(-total)][1:10, event_name]
plot_data <- action_summary[event_name %in% top_actions]

# Plot: Top 10 Actions by Outcome
ggplot(plot_data, aes(x = reorder(event_name, avg_count), y = avg_count, fill = final_outcome)) +
  geom_col(position = "dodge") +
  coord_flip() +
  scale_fill_manual(values = outcome_colors) +
  labs(
    title = "Top 10 Most Common Actions by Journey Result",
    x = "Action Type",
    y = "Average Times Per User"
  ) +
  theme_minimal()

ggsave("charlie/eda-plots/top_actions_by_outcome.png", width = 8, height = 6)
