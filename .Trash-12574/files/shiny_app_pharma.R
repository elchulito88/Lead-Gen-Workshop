# This is a Shiny web application. You can run the application by clicking
# the 'Run App' button above.
#
# Find out more about building applications with Shiny here:
#
#    http://shiny.rstudio.com/
#
install.packages("png")

library(shiny)
library(png)
library(httr)
library(jsonlite)
library(plotly)
library(ggplot2)

# Define UI for application that draws a histogram
ui <- fluidPage(
  
  # Application title
  titlePanel("Drug Efficacy Prediction"),
  
  # Sidebar with inputs for drug properties 
  sidebarLayout(
    sidebarPanel(
      numericInput(inputId="feat1",
                   label='pH Level', 
                   value=7),
      numericInput(inputId="feat2",
                   label='Molecular Weight', 
                   value=500),
      numericInput(inputId="feat3",
                   label='Solubility', 
                   value=0.01),
      numericInput(inputId="feat4",
                   label='Bioavailability', 
                   value=0.7),
      numericInput(inputId="feat5",
                   label='Dosage (mg)', 
                   value=50),
      actionButton("predict", "Predict")
    ),
    
    # Show a plot of the generated distribution
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

prediction <- function(inpFeat1,inpFeat2,inpFeat3,inpFeat4,inpFeat5) {
  
#### Ensure you replace this with appropriate API call or model prediction logic ####
url <- "https://se-demo.domino.tech:443/models/65f1aa2fe1c0e22bf72279cb/latest/model"
response <- POST(
  url,
 authenticate("1rH1VskniA4xeLonhleJNIKES5iPnZ80ciqu9oLfKySkQduk6oS4U1CxaKe4hFSP", "1rH1VskniA4xeLonhleJNIKES5iPnZ80ciqu9oLfKySkQduk6oS4U1CxaKe4hFSP", type = "basic"),   
    body=toJSON(list(data=list(pH_level = inpFeat1, 
                               molecular_weight = inpFeat2,
                               solubility = inpFeat3,
                               bioavailability = inpFeat4,
                               dosage_mg = inpFeat5)), auto_unbox = TRUE),
    content_type("application/json")
  )
  
  str(content(response))
  
  result <- content(response)
}

gauge <- function(pos,breaks=c(0,2.5,5,7.5, 10)) {
 
  # Function and plot settings remain the same, as they're generic enough
  # for our new use case.
}

# Define server logic required to draw a histogram
server <- function(input, output,session) {
  
  observeEvent(input$predict, {
    updateTabsetPanel(session, "inTabset",
                      selected = paste0("pnlPredict", input$controller)
    )
    print(input)
    result <- prediction(input$feat1, input$feat2, input$feat3, input$feat4, input$feat5)
    print(result)
    
    pred <- result$result[[1]][[1]]
    modelVersion <- result$release$model_version_number
    responseTime <- result$model_time_in_ms
    output$summary <- renderText({paste0("Drug Efficacy estimate is ", round(pred,2))})
    output$version <- renderText({paste0("Model version used for scoring : ", modelVersion)})
    output$reponsetime <- renderText({paste0("Model response time : ", responseTime, " ms")})
    output$plot <- renderPlotly({
      gauge(round(pred,2))
    })
  })
  
}

# Run the application 
shinyApp(ui = ui, server = server)