library(mlflow)
library(jsonlite)

print("Reading in data")
project_name <- Sys.getenv('DOMINO_PROJECT_NAME')
path <- paste('/mnt/data/mlops-best-practices/credit_card_default.csv')
path <- gsub(" ", "", path, fixed = TRUE)
data <- read.csv(file = path)
head(data)

# Verify the renaming
print("Columns in data after renaming:")
print(colnames(data))

# Define MLflow experiment
mlflow_set_experiment(experiment_name = paste0(Sys.getenv('DOMINO_PROJECT_NAME'), " ", Sys.getenv('DOMINO_STARTING_USERNAME'), " ", Sys.getenv('MLFLOW_NAME')))

# Remove missing values
data <- na.omit(data)
print(paste("Number of rows with missing values removed:", dim(data)[1] - sum(complete.cases(data))))

# Split data into training and testing sets
set.seed(123)  # Set seed for reproducibility
train <- data[sample(nrow(data), round(dim(data)[1] * 0.75)), ]
test <- data[!(rownames(data) %in% rownames(train)), ]

# Verify that the train and test sets include the "DEFAULT" column
if (!("DEFAULT" %in% colnames(train))) {
  stop("Column 'DEFAULT' is not present in the training set.")
}

# Define target and feature columns
target_variable <- "DEFAULT"
features <- setdiff(names(data), target_variable)

train_matrix <- as.matrix(train[, features])
test_matrix <- as.matrix(test[, features])
label_matrix <- as.matrix(train[[target_variable]])
test_lab_matrix <- as.matrix(test[[target_variable]])

dim(train) + dim(test)

# Start MLflow run
with(mlflow_start_run(), {
  mlflow_set_tag("Model_Type", "R")
  print("Training Model")
  
  # Train the model (update formula for new dataset)
  lm_model <- lm(formula = as.formula(paste(target_variable, "~ .")), data = train)
  print(lm_model)
  
  # Define RSQUARE function
  RSQUARE <- function(y_actual, y_predict) {
    cor(y_actual, y_predict)^2
  }
  
  # Predict and calculate metrics
  preds_lm <- predict(lm_model, newdata = test)
  
  rsquared_lm <- round(RSQUARE(test[[target_variable]], preds_lm), 3)
  print(rsquared_lm)
  
  # Mean Squared Error
  mse_lm <- round(mean((test_lab_matrix - preds_lm)^2), 3)
  print(mse_lm)
  
  # Log metrics to MLflow
  mlflow_log_metric("R2", rsquared_lm)
  mlflow_log_metric("MSE", mse_lm)
  
  # Save diagnostics to JSON
  diagnostics <- list("R2" = rsquared_lm, "MSE" = mse_lm)
  fileConn <- file("dominostats.json")
  writeLines(toJSON(diagnostics), fileConn)
  close(fileConn)
  
  # Save model
  save(lm_model, file = "/mnt/code/models/R_linear_model.Rda")
})