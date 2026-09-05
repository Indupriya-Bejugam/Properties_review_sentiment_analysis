import matplotlib.pyplot as plt
from textblob import TextBlob

# Reviews to analyze
reviews = [
    "Living at Harmony Residences in Manikonda has been a wonderful experience. The apartments are beautifully designed with modern fittings and spacious layouts. The community is very friendly and there are numerous activities organized for both children and adults. The location is convenient with excellent connectivity to key areas in Hyderabad, and there are plenty of shopping and dining options nearby. The maintenance staff is efficient and the security is top-notch, making it a safe and pleasant place to live.",
    "Crystal Gardens in Hanamkonda is an excellent choice for families. The apartments are well-constructed with a focus on quality and comfort. The complex is surrounded by lush greenery, creating a serene environment that is hard to find in the city. The amenities, including the clubhouse and gym, are well-maintained and the management is very responsive to any issues. The proximity to good schools and hospitals adds to the convenience of living here.",
    "Living in Prestige Enclave in Sainikpuri has been a joy. The apartments are spacious and well-designed, providing a comfortable living environment. The community is vibrant and friendly, with various activities and events bringing residents together. The locality is quiet and green, which is perfect for families looking for a peaceful place to live. The security measures are thorough, and the maintenance staff is prompt and efficient, ensuring that the property is always in excellent condition.",
    "Pearl Heights in Malkajgiri offers an exceptional living experience. The apartments are modern and equipped with all the necessary amenities. The complex includes a well-maintained gym, swimming pool, and children's play area. The management team is professional and always ready to assist with any issues. The location is convenient, with easy access to public transportation, schools, and shopping centers. It's a great place for families and working professionals alike.",
    "Serene Villas in Maheshwaram are an excellent choice for those seeking tranquility and luxury. The villas are spacious, with beautiful interiors and well-manicured gardens. The community is well-secured, providing a safe environment for residents. The amenities, including a clubhouse, sports facilities, and a jogging track, are top-notch. The locality is rapidly developing, with new amenities and infrastructure being added regularly, making it a promising investment.",
    "Living in Coral Residency in Alwal has been quite disappointing. The apartments suffer from poor construction quality, with frequent plumbing and electrical issues. The noise from nearby construction sites and traffic makes it difficult to enjoy any peace and quiet. The management is unresponsive and slow to address residents' concerns, which adds to the frustration. The locality also lacks good schools and hospitals, making it an inconvenient place for families.",
    "My experience at Emerald Towers in Boduppal has been far from satisfactory. The apartments are cramped and poorly ventilated, making them uncomfortable to live in. There are persistent issues with water supply and the maintenance staff is neither efficient nor prompt. The surrounding area is underdeveloped, with limited access to basic amenities like grocery stores and medical facilities. The overall living experience has been quite negative.",
    "Staying at Sunflower Apartments in Bolarum has been a frustrating ordeal. The building is old and poorly maintained, with frequent issues like water leaks and mold. The elevators often break down, and the common areas are dirty and neglected. The locality is noisy and overcrowded, with inadequate parking facilities adding to the inconvenience. The management's lack of responsiveness to residents' complaints only worsens the situation.",
    "Living in Palm Grove Residences in Kompally has been a refreshing experience. The apartments are spacious and come with modern fittings. The community is well-planned with plenty of green spaces, which makes it a great place for families with children. The security is excellent, ensuring a safe environment for everyone. The location is also convenient, with good schools, hospitals, and shopping centers nearby, making daily life quite comfortable.",
    "Golden Meadows in Kukatpally is a fantastic place to live. The apartments are well-constructed with ample natural light and ventilation. The complex boasts several amenities, including a well-equipped gym, a swimming pool, and a children's play area. The maintenance staff is responsive and keeps the property in excellent condition. The area is well-connected by public transport, and there are plenty of shops and restaurants within walking distance."
]

# Sentiment analysis
sentiments = [TextBlob(review).sentiment.polarity for review in reviews]
sentiment_labels = ['Positive' if sentiment > 0 else 'Negative' for sentiment in sentiments]

# Plotting the sentiment analysis
plt.figure(figsize=(10, 5))
bars = plt.bar(range(len(reviews)), sentiments, color=['green' if sentiment > 0 else 'red' for sentiment in sentiments])
plt.axhline(0, color='gray', linewidth=0.8)
plt.xlabel('Review Index')
plt.ylabel('Sentiment Polarity')
plt.title('Sentiment Analysis of Real Estate Reviews')

# Adding text labels
for bar, label in zip(bars, sentiment_labels):
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2 - 0.1, yval + 0.02 if yval >= 0 else yval - 0.05, label, color='black', va='center')

plt.xticks(range(len(reviews)), range(1, len(reviews)+1))
plt.ylim(-1, 1)
plt.show()