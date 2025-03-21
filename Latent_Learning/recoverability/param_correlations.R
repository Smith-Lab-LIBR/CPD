library(ggplot2)

# Define the file names (update these paths as needed)
file1 <- "L:/rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/threshold/2params/individual_single_inference_expectation.csv"  # CSV with parameter values (set 1)
file2 <- "L:/rsmith/lab-members/rhodson/CPD/CPD_results/latent_model/recoverability/2params/individual_latent_single_inference_expectation_recov.csv"   # CSV with parameter values (set 2)

# Read the CSV files
data1 <- read.csv(file1, stringsAsFactors = FALSE)
data2 <- read.csv(file2, stringsAsFactors = FALSE)

# Define a function to plot the correlation between latent_lr and new_latent_lr

#-------------------------------------------------------------------
# Define the list of parameters to analyze
parameters <- c("reward_lr", "latent_lr", "reward_prior", "inverse_temp")

#-------------------------------------------------------------------
# Function to plot correlation between any two parameters for a given dataset.
plotCorrelationPair <- function(data, param1, param2, fileLabel) {
  # Check if required columns exist in the dataset
  if (!all(c(param1, param2) %in% names(data))) {
    stop(sprintf("Column %s or %s is missing in the dataset for %s.", 
                 param1, param2, fileLabel))
  }
  
  # Perform the Pearson correlation test
  cor_result <- cor.test(data[[param1]], data[[param2]])
  
  # Create the scatter plot with a linear regression line
  p <- ggplot(data, aes_string(x = param1, y = param2)) +
    geom_point(size = 2, alpha = 0.7) +
    geom_smooth(method = "lm", se = FALSE, color = "blue") +
    labs(
      title = paste("Correlation between", param1, "and", param2, "in", fileLabel),
      subtitle = paste("r =", round(cor_result$estimate, 3),
                       ", p =", signif(cor_result$p.value, 3)),
      x = param1,
      y = param2
    ) +
    theme_minimal()
  
  print(p)
}

#-------------------------------------------------------------------
# Generate all pairwise combinations of the parameters using combn()
paramPairs <- combn(parameters, 2, simplify = FALSE)

#-------------------------------------------------------------------
# For each file, loop over all pairs and plot the correlations

# For File 1
cat("Generating correlation plots for File 1...\n")
for (pair in paramPairs) {
  plotCorrelationPair(data1, pair[1], pair[2], "File 1")
}

# For File 2
cat("Generating correlation plots for File 2...\n")
for (pair in paramPairs) {
  plotCorrelationPair(data2, pair[1], pair[2], "File 2")
}
