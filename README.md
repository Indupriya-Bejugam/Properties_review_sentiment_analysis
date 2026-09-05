This is my first ML application and it is on the basis of a field close to my dad, that's real-estate.
The properties reviews sentiment analysis is a project which is based on natural language processing, where we use NLP techniques to extract useful words of each review and based on these words we can use binary classification to predict the property sentiment if it's positive or negative. More to build. Updating soon.

<img width="1400" height="764" alt="image" src="https://github.com/user-attachments/assets/80ae0aa3-2899-4697-839a-805aac7b7618" />

New updations:

                  PROPERTY WEBSITES
                  /              \
                 /                \
        Property Listings        Reviews
              ↓                    ↓
        Web Scraping           Text Cleaning
        BeautifulSoup               ↓
              ↓                  NLP
        Property Data               ↓
              ↓              Sentiment Model
        Preprocessing                ↓
              ↓             Positive/Negative
           TF-IDF                    ↓
              ↓               Sentiment Score
       Property Features             │
              ↓                      │
          K-Means                    │
              ↓                      │
        Similar Properties            │
              \                     /
               \                   /
                ↓                 ↓
             RECOMMENDATION ENGINE
                     ↓
              User Preferences
                     ↓
               Ranking / Score
                     ↓
                  Flask
                     ↓
                Web Interface

<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/091392bc-8502-416f-9c4f-4de38eab4cb0" />

