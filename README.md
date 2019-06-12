### Hearthstone Arena Card Classifier

Hearthstone is a game of skill, chance, and theory crafting. The idea of the game is to reduce your opponent's HP to 0.
We are focusing specifically on Heartstone's Arena game mode, a buy-in type of tournament where players are given random classes and cards to choose from. Some cards are good, and some are really bad, but to a novice player, everycard is confusing to them. That is why we created a program to help them choose the best cards to maximize their chances of winning.

Our three classifiers (KNN, Naive Bayes, and Decision Tree) all aim to classify how well a card does given its statistics that can be measures. Namely, Mana cost, Mechanic Text, Card Type, Mechanic Type, Bold Words, Heath, and Attack. Given these seven features, we try and determine the merit of the card.

### RUN

How to run:

```
make
or
make run
```

This will run all of our classifiers in this order: KNN, Naive Bayes, Decision Tree. A graph will show up for each classifier, so in order to continue execution, just close the graph.
