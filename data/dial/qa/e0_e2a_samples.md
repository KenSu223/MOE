# e0_e2a: 50 seeded samples (qwen-family texts; random_state 0)

Each item: prompt A -> answer A | prompt B -> answer B (STR at token positions str_pos). Orders shown where built.

## 1. ctrl|party|funeral|happy|sad (objects, ext)
- order 1: `'Of the party and the funeral, the party was'` -> **happy**
  - B: `'Of the party and the funeral, the funeral was'` -> **sad**
- order 2: `'Of the funeral and the party, the party was'` -> **happy**
  - B: `'Of the funeral and the party, the funeral was'` -> **sad**

## 2. ctrl|birthday|funeral|happy|sad (objects, ext)
- order 1: `'Of the birthday and the funeral, the birthday was'` -> **happy**
  - B: `'Of the birthday and the funeral, the funeral was'` -> **sad**
- order 2: `'Of the funeral and the birthday, the birthday was'` -> **happy**
  - B: `'Of the funeral and the birthday, the funeral was'` -> **sad**

## 3. ctrl|cloud|mud|high|low (objects, ext)
- order 1: `'Of the cloud and the mud, the cloud was'` -> **high**
  - B: `'Of the cloud and the mud, the mud was'` -> **low**
- order 2: `'Of the mud and the cloud, the cloud was'` -> **high**
  - B: `'Of the mud and the cloud, the mud was'` -> **low**

## 4. ctrl|stone|wool|hard|soft (objects, ext)
- order 1: `'Of the stone and the wool, the stone was'` -> **hard**
  - B: `'Of the stone and the wool, the wool was'` -> **soft**
- order 2: `'Of the wool and the stone, the stone was'` -> **hard**
  - B: `'Of the wool and the stone, the wool was'` -> **soft**

## 5. ctrl|song|funeral|happy|sad (objects, ext)
- order 1: `'Of the song and the funeral, the song was'` -> **happy**
  - B: `'Of the song and the funeral, the funeral was'` -> **sad**
- order 2: `'Of the funeral and the song, the song was'` -> **happy**
  - B: `'Of the funeral and the song, the funeral was'` -> **sad**

## 6. ctrl|vacation|commute|fun|boring (objects, ext)
- order 1: `'Of the vacation and the commute, the vacation was'` -> **fun**
  - B: `'Of the vacation and the commute, the commute was'` -> **boring**
- order 2: `'Of the commute and the vacation, the vacation was'` -> **fun**
  - B: `'Of the commute and the vacation, the commute was'` -> **boring**

## 7. ctrl|stove|freezer|hot|cold (objects, ext)
- order 1: `'Of the stove and the freezer, the stove was'` -> **hot**
  - B: `'Of the stove and the freezer, the freezer was'` -> **cold**
- order 2: `'Of the freezer and the stove, the stove was'` -> **hot**
  - B: `'Of the freezer and the stove, the freezer was'` -> **cold**

## 8. ctrl|salsa|yogurt|spicy|mild (objects, ext)
- order 1: `'Of the salsa and the yogurt, the salsa was'` -> **spicy**
  - B: `'Of the salsa and the yogurt, the yogurt was'` -> **mild**
- order 2: `'Of the yogurt and the salsa, the salsa was'` -> **spicy**
  - B: `'Of the yogurt and the salsa, the yogurt was'` -> **mild**

## 9. ctrl|concert|library|loud|quiet (objects, ext)
- order 1: `'Of the concert and the library, the concert was'` -> **loud**
  - B: `'Of the concert and the library, the library was'` -> **quiet**
- order 2: `'Of the library and the concert, the concert was'` -> **loud**
  - B: `'Of the library and the concert, the library was'` -> **quiet**

## 10. ctrl|giant|child|tall|short (objects, ext)
- order 1: `'Of the giant and the child, the giant was'` -> **tall**
  - B: `'Of the giant and the child, the child was'` -> **short**
- order 2: `'Of the child and the giant, the giant was'` -> **tall**
  - B: `'Of the child and the giant, the child was'` -> **short**

## 11. ctrl|hospital|dump|clean|dirty (objects, ext)
- order 1: `'Of the hospital and the dump, the hospital was'` -> **clean**
  - B: `'Of the hospital and the dump, the dump was'` -> **dirty**
- order 2: `'Of the dump and the hospital, the hospital was'` -> **clean**
  - B: `'Of the dump and the hospital, the dump was'` -> **dirty**

## 12. ctrl|candy|lemon|sweet|sour (objects, ext)
- order 1: `'Of the candy and the lemon, the candy was'` -> **sweet**
  - B: `'Of the candy and the lemon, the lemon was'` -> **sour**
- order 2: `'Of the lemon and the candy, the candy was'` -> **sweet**
  - B: `'Of the lemon and the candy, the lemon was'` -> **sour**

## 13. ctrl|lake|desert|wet|dry (objects, ext)
- order 1: `'Of the lake and the desert, the lake was'` -> **wet**
  - B: `'Of the lake and the desert, the desert was'` -> **dry**
- order 2: `'Of the desert and the lake, the lake was'` -> **wet**
  - B: `'Of the desert and the lake, the desert was'` -> **dry**

## 14. ctrl|knife|spoon|sharp|dull (objects, ext)
- order 1: `'Of the knife and the spoon, the knife was'` -> **sharp**
  - B: `'Of the knife and the spoon, the spoon was'` -> **dull**
- order 2: `'Of the spoon and the knife, the knife was'` -> **sharp**
  - B: `'Of the spoon and the knife, the spoon was'` -> **dull**

## 15. ctrl|soup|ice|hot|cold (objects, ext)
- order 1: `'Of the soup and the ice, the soup was'` -> **hot**
  - B: `'Of the soup and the ice, the ice was'` -> **cold**
- order 2: `'Of the ice and the soup, the soup was'` -> **hot**
  - B: `'Of the ice and the soup, the ice was'` -> **cold**

## 16. ctrl|bear|rabbit|big|small (objects, ext)
- order 1: `'Of the bear and the rabbit, the bear was'` -> **big**
  - B: `'Of the bear and the rabbit, the rabbit was'` -> **small**
- order 2: `'Of the rabbit and the bear, the bear was'` -> **big**
  - B: `'Of the rabbit and the bear, the rabbit was'` -> **small**

## 17. ctrl|egg|rock|fragile|sturdy (objects, ext)
- order 1: `'Of the egg and the rock, the egg was'` -> **fragile**
  - B: `'Of the egg and the rock, the rock was'` -> **sturdy**
- order 2: `'Of the rock and the egg, the egg was'` -> **fragile**
  - B: `'Of the rock and the egg, the rock was'` -> **sturdy**

## 18. ctrl|party|lecture|fun|boring (objects, ext)
- order 1: `'Of the party and the lecture, the party was'` -> **fun**
  - B: `'Of the party and the lecture, the lecture was'` -> **boring**
- order 2: `'Of the lecture and the party, the party was'` -> **fun**
  - B: `'Of the lecture and the party, the lecture was'` -> **boring**

## 19. ctrl|soap|dirt|clean|dirty (objects, ext)
- order 1: `'Of the soap and the dirt, the soap was'` -> **clean**
  - B: `'Of the soap and the dirt, the dirt was'` -> **dirty**
- order 2: `'Of the dirt and the soap, the soap was'` -> **clean**
  - B: `'Of the dirt and the soap, the dirt was'` -> **dirty**

## 20. ctrl|sword|stick|sharp|blunt (objects, ext)
- order 1: `'Of the sword and the stick, the sword was'` -> **sharp**
  - B: `'Of the sword and the stick, the stick was'` -> **blunt**
- order 2: `'Of the stick and the sword, the sword was'` -> **sharp**
  - B: `'Of the stick and the sword, the stick was'` -> **blunt**

## 21. ctrl|kale|cookie|bitter|sweet (objects, ext)
- order 1: `'Of the kale and the cookie, the kale was'` -> **bitter**
  - B: `'Of the kale and the cookie, the cookie was'` -> **sweet**
- order 2: `'Of the cookie and the kale, the kale was'` -> **bitter**
  - B: `'Of the cookie and the kale, the cookie was'` -> **sweet**

## 22. ctrl|movie|exam|fun|boring (objects, ext)
- order 1: `'Of the movie and the exam, the movie was'` -> **fun**
  - B: `'Of the movie and the exam, the exam was'` -> **boring**
- order 2: `'Of the exam and the movie, the movie was'` -> **fun**
  - B: `'Of the exam and the movie, the exam was'` -> **boring**

## 23. ctrl|river|path|wide|narrow (objects, ext)
- order 1: `'Of the river and the path, the river was'` -> **wide**
  - B: `'Of the river and the path, the path was'` -> **narrow**
- order 2: `'Of the path and the river, the river was'` -> **wide**
  - B: `'Of the path and the river, the path was'` -> **narrow**

## 24. 3O2Y2UIUCQSZRZPVQ54P4YAQ7QOFKP (objects, ext)
- order 1: `'Of the bedroom and the house, the bedroom was too'` -> **large**
  - B: `'Of the bedroom and the house, the house was too'` -> **small**
- order 2: `'Of the house and the bedroom, the bedroom was too'` -> **large**
  - B: `'Of the house and the bedroom, the house was too'` -> **small**

## 25. ctrl|carnival|meeting|fun|boring (objects, ext)
- order 1: `'Of the carnival and the meeting, the carnival was'` -> **fun**
  - B: `'Of the carnival and the meeting, the meeting was'` -> **boring**
- order 2: `'Of the meeting and the carnival, the carnival was'` -> **fun**
  - B: `'Of the meeting and the carnival, the meeting was'` -> **boring**

## 26. ctrl|tea|snow|hot|cold (objects, ext)
- order 1: `'Of the tea and the snow, the tea was'` -> **hot**
  - B: `'Of the tea and the snow, the snow was'` -> **cold**
- order 2: `'Of the snow and the tea, the tea was'` -> **hot**
  - B: `'Of the snow and the tea, the snow was'` -> **cold**

## 27. ctrl|road|path|long|short (objects, ext)
- order 1: `'Of the road and the path, the road was'` -> **long**
  - B: `'Of the road and the path, the path was'` -> **short**
- order 2: `'Of the path and the road, the road was'` -> **long**
  - B: `'Of the path and the road, the path was'` -> **short**

## 28. ctrl|snow|mud|clean|dirty (objects, ext)
- order 1: `'Of the snow and the mud, the snow was'` -> **clean**
  - B: `'Of the snow and the mud, the mud was'` -> **dirty**
- order 2: `'Of the mud and the snow, the snow was'` -> **clean**
  - B: `'Of the mud and the snow, the mud was'` -> **dirty**

## 29. 38DCH97KHJGG1T8T6YR77G20O6RQJ5 (objects, ext)
- order 1: `'Of the cake and the card, the cake was'` -> **delicious**
  - B: `'Of the cake and the card, the card was'` -> **sentimental**
- order 2: `'Of the card and the cake, the cake was'` -> **delicious**
  - B: `'Of the card and the cake, the card was'` -> **sentimental**

## 30. ctrl|planet|marble|big|small (objects, ext)
- order 1: `'Of the planet and the marble, the planet was'` -> **big**
  - B: `'Of the planet and the marble, the marble was'` -> **small**
- order 2: `'Of the marble and the planet, the planet was'` -> **big**
  - B: `'Of the marble and the planet, the marble was'` -> **small**

## 31. ctrl|rock|feather|heavy|light (objects, ext)
- order 1: `'Of the rock and the feather, the rock was'` -> **heavy**
  - B: `'Of the rock and the feather, the feather was'` -> **light**
- order 2: `'Of the feather and the rock, the rock was'` -> **heavy**
  - B: `'Of the feather and the rock, the feather was'` -> **light**

## 32. ctrl|cow|rat|big|small (objects, ext)
- order 1: `'Of the cow and the rat, the cow was'` -> **big**
  - B: `'Of the cow and the rat, the rat was'` -> **small**
- order 2: `'Of the rat and the cow, the cow was'` -> **big**
  - B: `'Of the rat and the cow, the rat was'` -> **small**

## 33. ctrl|jet|boat|fast|slow (objects, ext)
- order 1: `'Of the jet and the boat, the jet was'` -> **fast**
  - B: `'Of the jet and the boat, the boat was'` -> **slow**
- order 2: `'Of the boat and the jet, the jet was'` -> **fast**
  - B: `'Of the boat and the jet, the boat was'` -> **slow**

## 34. ctrl|wolf|sheep|dangerous|safe (objects, ext)
- order 1: `'Of the wolf and the sheep, the wolf was'` -> **dangerous**
  - B: `'Of the wolf and the sheep, the sheep was'` -> **safe**
- order 2: `'Of the sheep and the wolf, the wolf was'` -> **dangerous**
  - B: `'Of the sheep and the wolf, the sheep was'` -> **safe**

## 35. ctrl|tea|ice|warm|cold (objects, ext)
- order 1: `'Of the tea and the ice, the tea was'` -> **warm**
  - B: `'Of the tea and the ice, the ice was'` -> **cold**
- order 2: `'Of the ice and the tea, the tea was'` -> **warm**
  - B: `'Of the ice and the tea, the ice was'` -> **cold**

## 36. ctrl|steel|cotton|hard|soft (objects, ext)
- order 1: `'Of the steel and the cotton, the steel was'` -> **hard**
  - B: `'Of the steel and the cotton, the cotton was'` -> **soft**
- order 2: `'Of the cotton and the steel, the steel was'` -> **hard**
  - B: `'Of the cotton and the steel, the cotton was'` -> **soft**

## 37. ctrl|palace|cabin|big|small (objects, ext)
- order 1: `'Of the palace and the cabin, the palace was'` -> **big**
  - B: `'Of the palace and the cabin, the cabin was'` -> **small**
- order 2: `'Of the cabin and the palace, the palace was'` -> **big**
  - B: `'Of the cabin and the palace, the cabin was'` -> **small**

## 38. ctrl|car|bike|fast|slow (objects, ext)
- order 1: `'Of the car and the bike, the car was'` -> **fast**
  - B: `'Of the car and the bike, the bike was'` -> **slow**
- order 2: `'Of the bike and the car, the car was'` -> **fast**
  - B: `'Of the bike and the car, the bike was'` -> **slow**

## 39. ctrl|ocean|stream|wide|narrow (objects, ext)
- order 1: `'Of the ocean and the stream, the ocean was'` -> **wide**
  - B: `'Of the ocean and the stream, the stream was'` -> **narrow**
- order 2: `'Of the stream and the ocean, the ocean was'` -> **wide**
  - B: `'Of the stream and the ocean, the stream was'` -> **narrow**

## 40. ctrl|castle|hut|big|small (objects, ext)
- order 1: `'Of the castle and the hut, the castle was'` -> **big**
  - B: `'Of the castle and the hut, the hut was'` -> **small**
- order 2: `'Of the hut and the castle, the castle was'` -> **big**
  - B: `'Of the hut and the castle, the hut was'` -> **small**

## 41. ctrl|iron|velvet|hard|soft (objects, ext)
- order 1: `'Of the iron and the velvet, the iron was'` -> **hard**
  - B: `'Of the iron and the velvet, the velvet was'` -> **soft**
- order 2: `'Of the velvet and the iron, the iron was'` -> **hard**
  - B: `'Of the velvet and the iron, the velvet was'` -> **soft**

## 42. ctrl|lake|dust|wet|dry (objects, ext)
- order 1: `'Of the lake and the dust, the lake was'` -> **wet**
  - B: `'Of the lake and the dust, the dust was'` -> **dry**
- order 2: `'Of the dust and the lake, the lake was'` -> **wet**
  - B: `'Of the dust and the lake, the dust was'` -> **dry**

## 43. ctrl|horse|ant|big|small (objects, ext)
- order 1: `'Of the horse and the ant, the horse was'` -> **big**
  - B: `'Of the horse and the ant, the ant was'` -> **small**
- order 2: `'Of the ant and the horse, the horse was'` -> **big**
  - B: `'Of the ant and the horse, the ant was'` -> **small**

## 44. ctrl|ocean|desert|wet|dry (objects, ext)
- order 1: `'Of the ocean and the desert, the ocean was'` -> **wet**
  - B: `'Of the ocean and the desert, the desert was'` -> **dry**
- order 2: `'Of the desert and the ocean, the ocean was'` -> **wet**
  - B: `'Of the desert and the ocean, the desert was'` -> **dry**

## 45. ctrl|stadium|graveyard|full|empty (objects, ext)
- order 1: `'Of the stadium and the graveyard, the stadium was'` -> **full**
  - B: `'Of the stadium and the graveyard, the graveyard was'` -> **empty**
- order 2: `'Of the graveyard and the stadium, the stadium was'` -> **full**
  - B: `'Of the graveyard and the stadium, the graveyard was'` -> **empty**

## 46. ctrl|road|path|wide|narrow (objects, ext)
- order 1: `'Of the road and the path, the road was'` -> **wide**
  - B: `'Of the road and the path, the path was'` -> **narrow**
- order 2: `'Of the path and the road, the road was'` -> **wide**
  - B: `'Of the path and the road, the path was'` -> **narrow**

## 47. ctrl|train|wagon|fast|slow (objects, ext)
- order 1: `'Of the train and the wagon, the train was'` -> **fast**
  - B: `'Of the train and the wagon, the wagon was'` -> **slow**
- order 2: `'Of the wagon and the train, the train was'` -> **fast**
  - B: `'Of the wagon and the train, the wagon was'` -> **slow**

## 48. 3HYV4299H2A2K9M563ZGO948PS98ER (objects, core)
- order 1: `'Of the head and the crowd, the head was'` -> **bald**
  - B: `'Of the head and the crowd, the crowd was'` -> **vast**
- order 2: `'Of the crowd and the head, the head was'` -> **bald**
  - B: `'Of the crowd and the head, the crowd was'` -> **vast**

## 49. ctrl|airplane|bee|large|tiny (objects, ext)
- order 1: `'Of the airplane and the bee, the airplane was'` -> **large**
  - B: `'Of the airplane and the bee, the bee was'` -> **tiny**
- order 2: `'Of the bee and the airplane, the airplane was'` -> **large**
  - B: `'Of the bee and the airplane, the bee was'` -> **tiny**

## 50. ctrl|highway|graveyard|noisy|silent (objects, ext)
- order 1: `'Of the highway and the graveyard, the highway was'` -> **noisy**
  - B: `'Of the highway and the graveyard, the graveyard was'` -> **silent**
- order 2: `'Of the graveyard and the highway, the highway was'` -> **noisy**
  - B: `'Of the graveyard and the highway, the graveyard was'` -> **silent**
