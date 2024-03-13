install.packages("png")

library(shiny)
library(png)
library(httr)
library(jsonlite)
library(plotly)
library(ggplot2)

# Define UI for application that predicts drug efficacy
ui <- fluidPage(
  
  # Application title
  titlePanel("Drug Efficacy Predictor"),
  
  # Sidebar with inputs for prediction features
  sidebarLayout(
    sidebarPanel(
      numericInput(inputId="feat1",
                   label='Drug Dosage (mg)', 
                   value=100),
      numericInput(inputId="feat2",
                   label='Patient Age (years)', 
                   value=45),
      numericInput(inputId="feat3",
                   label='Disease Severity (0-10 scale)', 
                   value=5),
      selectInput(inputId="feat4",
                  label='Treatment Type',
                  choices=c("Type 1"=1, "Type 2"=2, "Type 3"=3),
                  selected=1),
      numericInput(inputId="feat5",
                   label='Body Mass Index (BMI)', 
                   value=25),
      actionButton("predict", "Predict")
    ),
    
    # Show a plot and summary of the prediction
    mainPanel(
      tabsetPanel(id = "inTabset", type = "tabs",
                  tabPanel(title="Prediction",value = "pnlPredict",
                           plotlyOutput("plot"),
                           verbatimTextOutput("summary"),
                           verbatimTextOutput("version"),
                           verbatimTextOutput("reponsetime"))
      )        
    )
  )
)

# Replace the prediction function with one relevant to your pharmaceutical model
prediction <- function(inpFeat1,inpFeat2,inpFeat3,inpFeat4,inpFeat5) {
  
  # You should replace the URL and authentication details with those of your actual model
url <- "https://se-demo.domino.tech:443/models/65f1aa2fe1c0e22bf72279cb/latest/model"
response <- POST(
  url,
 authenticate("1rH1VskniA4xeLonhleJNIKES5iPnZ80ciqu9oLfKySkQduk6oS4U1CxaKe4hFSP", "1rH1VskniA4xeLonhleJNIKES5iPnZ80ciqu9oLfKySkQduk6oS4U1CxaKe4hFSP", type = "basic"),  
    body=toJSON(list(data=list(dosage = inpFeat1, 
                               age = inpFeat2,
                               severity = inpFeat3,
                               treatment_type = inpFeat4,
                               bmi = inpFeat5)), auto_unbox = TRUE),
    content_type("application/json")
  )
  
  result <- content(response)
}

# Optionally, adjust the gauge function to reflect the context of drug efficacy or patient outcome prediction

# Define server logic
server <- function(input, output,session) {
  
  observeEvent(input$predict, {
    updateTabsetPanel(session, "inTabset",
                      selected = paste0("pnlPredict", input$controller)
    )
    
    result <- prediction(input$feat1, input$feat2, input$feat3, input$feat4, input$feat5)
    
    pred <- result$result[[1]][[1]]
    modelVersion <- result$release$model_version_number
    responseTime <- result$model_time_in_ms
    output$summary <- renderText({paste0("Predicted Efficacy: ", round(pred,2), " (0-100 scale)")})
    output$version <- renderText({paste0("Model version: ", modelVersion)})
    output$reponsetime <- renderText({paste0("Response time: ", responseTime, " ms")})
    output$plot <- renderPlotly({
      # Adjust the plot if necessary to better represent your prediction's context
      gauge(round(pred,2))
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)
