# read the library (rvest)
library(rvest)

#get all links: daily basis
p0001 <- read_html(unlist("https://www.borgerforslag.dk/"))

links <- p0001 %>% 
  html_nodes("a") %>% 
  html_attr("href")

links <- data.frame(links)

head(links)

links <- as.character(unique(links[grep("/se-og-stoet-forslag/", links$links),]))

links <- paste("https://www.borgerforslag.dk",links,sep="")

date <- rep(Sys.time(),length(links))
dailylinks <- data.frame(links,date)
colnames(dailylinks) = c("link","linkdate")

head(dailylinks)

#scraping all urls from the crawl ####
proposal <- c()
titles <- c()

for(url in links){
  page <- read_html(unlist(url))
  
  texts <- page %>% 
    html_node("div.article") %>%
    html_text() 
  
  titel <- page %>%
    html_nodes("div.cc552X") %>%
    html_text()
  
  titles <- append(titles, titel[1])
  proposal <- append(proposal, texts)
}

proposalData <- data.frame(titles,proposal)
