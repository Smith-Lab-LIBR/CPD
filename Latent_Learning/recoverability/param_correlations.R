library(ggplot2)

#-------------------------------------------------------------------
# Define the list of parameters to analyze
parameters <- c("reward_lr", "latent_lr", "num_states")

#-------------------------------------------------------------------
# Read the CSV files
file1 <- "L:/rsmith/lab-members/rhodson/CPD/CPD_results/combined/smaller_comp/CPD_latent_single_inference_expectation_new.csv"
file2 <- "L:/rsmith/lab-members/rhodson/CPD/CPD_results/combined/DDM/CPD_latent_single_inference_expectation_ddm_mapping3.csv"

#file1 <- "L:/rsmith/lab-members/rhodson/CPD/CPD_results/combined/smaller_comp/RW_single_basic.csv"
#file2 <- "L:/rsmith/lab-members/rhodson/CPD/CPD_results/combined/DDM/individual_CPD_RW_single_basic_ddm_mapping3.csv"

data1 <- read.csv(file1, stringsAsFactors = FALSE)
data2 <- read.csv(file2, stringsAsFactors = FALSE)

#-------------------------------------------------------------------
# Function to plot correlation between the same parameter across two datasets
plotCorrelationAcrossFiles <- function(data1, data2, param, fileLabel1, fileLabel2) {
  # Check if parameter exists in both datasets
  if (!param %in% names(data1) || !param %in% names(data2)) {
    stop(sprintf("Parameter %s is missing in one of the datasets.", param))
  }
  
  # Make sure data1 and data2 are aligned (same subjects/order)
  # You must verify this outside if needed — otherwise this just blindly pairs rows
  
  # Create a temporary dataframe for plotting
  temp_df <- data.frame(
    param_file1 = data1[[param]],
    param_file2 = data2[[param]]
  )
  
  # Perform Pearson correlation
  cor_result <- cor.test(temp_df$param_file1, temp_df$param_file2)
  
  # Create scatter plot
  p <- ggplot(temp_df, aes(x = param_file1, y = param_file2)) +
    geom_point(size = 2, alpha = 0.7) +
    geom_smooth(method = "lm", se = FALSE, color = "blue") +
    labs(
      title = paste("Correlation of", param, "between", fileLabel1, "and", fileLabel2),
      subtitle = paste("r =", round(cor_result$estimate, 3),
                       ", p =", signif(cor_result$p.value, 3)),
      x = paste(param, "-", fileLabel1),
      y = paste(param, "-", fileLabel2)
    ) +
    theme_minimal()
  
  print(p)
}

#-------------------------------------------------------------------
# Loop over parameters and plot correlations
cat("Generating cross-file correlation plots...\n")
for (param in parameters) {
  plotCorrelationAcrossFiles(data1, data2, param, "File 1", "File 2")
}

file3 <- "L:/rsmith/lab-members/rhodson/CPD/CPD_results/combined/DDM/CPD_latent_single_inference_expectation_ddm_mapping3.csv"  # <-- update the path
data3 <- read.csv(file3, stringsAsFactors = FALSE)

#-------------------------------------------------------------------
# List of parameters to correlate against num_states
parameters <- c("drift_baseline", "starting_bias", "drift_mod", "decision_thresh", "nondecision_time")

#-------------------------------------------------------------------
# Function to plot and test correlation
plotCorrelationNumStates <- function(data, param) {
  # Check if columns exist
  if (!all(c("reward_lr", param) %in% names(data))) {
    stop(sprintf("Column num_states or %s is missing.", param))
  }
  
  # Perform correlation
  cor_result <- cor.test(data$reward_lr, data[[param]])
  
  # Plot
  p <- ggplot(data, aes(x = reward_lr, y = .data[[param]])) +
    geom_point(size = 2, alpha = 0.7) +
    geom_smooth(method = "lm", se = FALSE, color = "blue") +
    labs(
      title = paste("Correlation between num_states and", param),
      subtitle = paste("r =", round(cor_result$estimate, 3),
                       ", p =", signif(cor_result$p.value, 3)),
      x = "reward_lr",
      y = param
    ) +
    theme_minimal()
  
  print(p)
}

#-------------------------------------------------------------------
# Loop over each parameter and plot
cat("Generating num_states correlation plots...\n")
for (param in parameters) {
  plotCorrelationNumStates(data3, param)
}
