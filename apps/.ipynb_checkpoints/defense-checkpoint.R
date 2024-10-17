# Load necessary libraries
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
  titlePanel("Lead Quality Prediction"),
  
  # Sidebar with a slider input for number of bins
  sidebarLayout(
    sidebarPanel(
      numericInput(inputId="feat1",
                   label='Circular Error Probable (CEP)', 
                   value=0.99),
      numericInput(inputId="feat2",
                   label='Mean Impact Point (MIP)', 
                   value=0.25),
      numericInput(inputId="feat3",
                   label='Miss Distance', 
                   value=0.05),
      numericInput(inputId="feat4",
                   label='Radial Error', 
                   value=1),
      numericInput(inputId="feat5",
                   label='Target Hit Probability (THP)', 
                   value=10),
      actionButton("predict", "Predict")
    ),
    
    # Show a plot of the generated distribution
    mainPanel(
      tabsetPanel(id = "inTabset", type = "tabs",
                  tabPanel(title="Prediction", value = "pnlPredict",
                           plotlyOutput("plot"),
                           verbatimTextOutput("summary"),
                           verbatimTextOutput("version"),
                           verbatimTextOutput("reponsetime"))
      )        
    )
  )
)

prediction <- function(inpFeat1, inpFeat2, inpFeat3, inpFeat4, inpFeat5) {
  # Ensure inputs are treated as numeric
  inpFeat1 <- as.numeric(inpFeat1)
  inpFeat2 <- as.numeric(inpFeat2)
  inpFeat3 <- as.numeric(inpFeat3)
  inpFeat4 <- as.numeric(inpFeat4)
  inpFeat5 <- as.numeric(inpFeat5)
  
  payload <- toJSON(list(
    data = list(
      density = sprintf("%.2f", as.numeric(inpFeat1)),
      volatile_acidity = sprintf("%.2f", as.numeric(inpFeat2)),
      chlorides = sprintf("%.2f", as.numeric(inpFeat3)),
      is_red = as.numeric(inpFeat4),  # Assuming this is intended to be an integer
      alcohol = sprintf("%.2f", as.numeric(inpFeat5))  # Forces float representation
    )
  ), auto_unbox = TRUE)
  
  #print("Payload JSON:")
  #print(payload)
  
  
  # Prepare JSON payload
  #payload <- toJSON(list(
  #  data = list(
  #    density = inpFeat1,
  #    volatile_acidity = inpFeat2,
  #    chlorides = inpFeat3,
  #    is_red = inpFeat4,
  #    alcohol = inpFeat5
  #  )
  #), auto_unbox = TRUE)
  
  # Print debugging information
  #print("Payload:")
  #print(payload)
  
url <- Sys.getenv("API_URL")
username <- Sys.getenv("API_USERNAME")
password <- Sys.getenv("API_PASSWORD")

response <- POST(
  url,
  authenticate(username, password, type = "basic"),
  body = payload,
  encode = "json",
  add_headers(`Content-Type` = "application/json")
  )
  
  # Print debugging information for response
  print("Response:")
  print(response)
  
  if (http_type(response) != "application/json") {
    stop("API did not return json")
  }
  
  result <- content(response, as = "parsed")
  
  # Print debugging information for result content
  print("Result:")
  print(result)
  
  return(result)
}

# Gauge plot function
gauge <- function(pos) {
  if (!is.finite(pos)) {
    stop("Error: Position must be a finite number")
  }
  
  breaks <- c(3, 7, 9, 10)  # Ensure the length of breaks covers the required ranges
  get.poly <- function(a, b, r1 = 0.5, r2 = 1.0) {
    if (!is.finite(a) || !is.finite(b)) {
      stop("Error: 'a' and 'b' must be finite numbers")
    }
    th.start <- pi * (1 - a / 10)
    th.end   <- pi * (1 - b / 10)
    th       <- seq(th.start, th.end, length = 10)
    x        <- c(r1 * cos(th), rev(r2 * cos(th)))
    y        <- c(r1 * sin(th), rev(r2 * sin(th)))
    return(data.frame(x, y))
  }
  ggplot() +
    geom_polygon(data = get.poly(breaks[1], breaks[2]), aes(x, y), fill = "red") +
    geom_polygon(data = get.poly(breaks[2], breaks[3]), aes(x, y), fill = "gold") +
    geom_polygon(data = get.poly(breaks[3], breaks[4]), aes(x, y), fill = "forestgreen") +
    geom_polygon(data = get.poly(pos - 0.2, pos + 0.2, 0.2), aes(x, y)) +
    geom_text(data = as.data.frame(breaks), size = 5, fontface = "bold", vjust = 0,
              aes(x = 1.1 * cos(pi * (1 - breaks / 10)), y = 1.1 * sin(pi * (1 - breaks / 10)), label = paste0(breaks))) +
    annotate("text", x = 0, y = 0, label = paste0(pos, " Points"), vjust = 0, size = 8, fontface = "bold") +
    coord_fixed() +
    theme_bw() +
    theme(axis.text = element_blank(),
          axis.title = element_blank(),
          axis.ticks = element_blank(),
          panel.grid = element_blank(),
          panel.border = element_blank())
}

# Define server logic required to draw a histogram
server <- function(input, output, session) {
  observeEvent(input$predict, {
    updateTabsetPanel(session, "inTabset",
                      selected = paste0("pnlPredict", input$controller)
    )
    result <- prediction(input$feat1, input$feat2, input$feat3, input$feat4, input$feat5)
    
    # Check if the result is valid
    if (is.null(result$result[[1]][[1]]) || is.na(result$result[[1]][[1]])) {
      output$summary <- renderText("Error: Invalid prediction result")
      return()
    }
    
    pred <- result$result[[1]][[1]]
    
    # Print prediction result for debugging
    print(paste("Prediction:", pred))
    
    if (!is.finite(pred)) {
      output$summary <- renderText("Error: Prediction result is not finite")
      return()
    }
    
    modelVersion <- result$release$model_version_number
    responseTime <- result$model_time_in_ms
    output$summary <- renderText({paste0("Quality estimate is ", round(pred, 2))})
    output$version <- renderText({paste0("Model version used for scoring: ", modelVersion)})
    output$reponsetime <- renderText({paste0("Model response time: ", responseTime, " ms")})
    output$plot <- renderPlotly({
      gauge(round(pred, 2))
    })
  })
}

# Run the application
shinyApp(ui = ui, server = server)
