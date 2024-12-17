# Load required libraries
library(caret)
library(randomForest)
library(xgboost)
library(dplyr)
library(ggplot2)

# Step 1: Load the Dataset (Example with synthetic data)
# Replace with the actual dataset path or use Criteo dataset
data <- read.csv("ctr_sample_data.csv")
head(data)

# Step 2: Data Preprocessing
data <- data %>%
  mutate(target = as.factor(target)) %>% # Convert target variable to factor
  select(-c(unwanted_column)) # Drop irrelevant columns if any

# Splitting the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$target, p = 0.8, list = FALSE)
trainData <- data[trainIndex,]
testData <- data[-trainIndex,]

# Step 3: Model Training using Random Forest
model_rf <- randomForest(target ~ ., data = trainData, ntree = 100, mtry = 3)
print(model_rf)

# Step 4: Model Training using XGBoost
train_matrix <- xgb.DMatrix(data = as.matrix(trainData[, -1]), label = as.numeric(trainData$target) - 1)
test_matrix <- xgb.DMatrix(data = as.matrix(testData[, -1]), label = as.numeric(testData$target) - 1)

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc"
)
model_xgb <- xgb.train(params, train_matrix, nrounds = 100, watchlist = list(train = train_matrix))

# Step 5: Model Evaluation
# Random Forest
pred_rf <- predict(model_rf, testData)
confusionMatrix(pred_rf, testData$target)

# XGBoost
pred_xgb <- predict(model_xgb, test_matrix)
auc <- roc(testData$target, pred_xgb)$auc
print(paste("AUC:", auc))

# Step 6: Visualization of Feature Importance (Random Forest)
importance <- varImp(model_rf)
print(importance)
plot(importance)

# Step 7: Deployment Example using Shiny
library(shiny)
ui <- fluidPage(
  titlePanel("CTR Prediction"),
  sidebarLayout(
    sidebarPanel(
      numericInput("feature1", "Feature 1:", value = 0),
      numericInput("feature2", "Feature 2:", value = 0)
    ),
    mainPanel(
      textOutput("prediction")
    )
  )
)
server <- function(input, output) {
  output$prediction <- renderText({
    new_data <- data.frame(feature1 = input$feature1, feature2 = input$feature2)
    pred <- predict(model_rf, new_data)
    paste("Predicted CTR:", pred)
  })
}
shinyApp(ui, server)
