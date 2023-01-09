# MusicRecommendationSystem
![image](docs/assets/img/DALL·E 2023-01-08 17.46.06 - an ai poet singing his song to a medival king.png " A cosmic Poet.")
## Introduction

Music is a form of art that has the ability to connect people across boundaries of space, time, and culture. It is also capable of evoking strong emotions and memories in listeners. In today's digital age, music is present in many aspects of daily life and social activity. The act of listening to music has become more complex due to the abundance of options available through large-scale music streaming services and the use of Big Data. By using these technologies and approaches, along with access to natural listening data gathered from various cultures, geographies, and economic settings, as well as song-level metadata, we can gain deeper insights into musical behavior on biological, social, and cultural levels. Through the final project of CS561, we will explore the domain of music recommendation systems, the inherent nature of the problem, and other interesting insights that we discover along the way.


## Philosophy of similarity

To construct a similarity metric for a recommendation system, we first need to consider the philosophy of similarity that the system will follow. There are two major types of recommendation systems: collaborative filtering systems and content-based systems. Collaborative filtering systems assume that if two users (x and y) are similar, then x is likely to enjoy the same things that y does. Content-based systems, on the other hand, consider the similarity between an item and what a user x likes, rather than looking at other similar users and their preferences.

There are also two approaches to determining what a user likes: implicit rating-based systems, which try to extract relevance and preferences from metadata generated by the user, and explicit rating systems, which simply ask the user directly. In this project, we will be using a collaborative filtering system that relies on an implicit rating system.

## Project Details

For this project, we plan to build a basic music recommendation system using the MLlib libraries that are part of the Spark installation. Our dataset will be the Million Song Dataset, which is a collection of audio features and metadata for one million contemporary popular music tracks. It does not include any audio, only derived features and metadata provided by The Echo Nest. We will also be using the Taste Profile dataset, which contains real user play counts for songs that have been matched to the Million Song Dataset. We will use the play counts as an implicit measure of a user's preference for a song, assuming that if a user has listened to a song more frequently compared to other songs, they like that song more. To choose the best hyperparameter combination, we will use a cross validator. Once we have the hyperparameters and rating matrix set, we will run Algorithm (Details in the full report) on the input matrix and try to fill in the empty cells. Then, we will return the top k rated songs as recommendations for the user.

## Interested In Knowing more?

This is part of a formal coursework hence this project is backed up with formal IEEE standard reports and presentations which are available in the repository in the report directory , The codebase is also available as a public repository in GitHub whose link is also available below. You can always contact me to know more or to discuss further ideas on the topic. 

## To run
All you need to do is to run the .py file , it will fire up the GUI Automatically.
