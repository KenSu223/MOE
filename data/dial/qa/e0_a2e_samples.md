# e0_a2e: 50 seeded samples (qwen-family texts; random_state 0)

Each item: prompt A -> answer A | prompt B -> answer B (STR at token positions str_pos). Orders shown where built.

## 1. ctrl|party|funeral|happy|sad (objects, ext)
- order 1: `'Of the party and the funeral. Fill in the blank: The _ was happy.\nAnswer: the'` -> **party**
  - B: `'Of the party and the funeral. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**
- order 2: `'Of the funeral and the party. Fill in the blank: The _ was happy.\nAnswer: the'` -> **party**
  - B: `'Of the funeral and the party. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**

## 2. ctrl|birthday|funeral|happy|sad (objects, ext)
- order 1: `'Of the birthday and the funeral. Fill in the blank: The _ was happy.\nAnswer: the'` -> **birthday**
  - B: `'Of the birthday and the funeral. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**
- order 2: `'Of the funeral and the birthday. Fill in the blank: The _ was happy.\nAnswer: the'` -> **birthday**
  - B: `'Of the funeral and the birthday. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**

## 3. ctrl|cloud|mud|high|low (objects, ext)
- order 1: `'Of the cloud and the mud. Fill in the blank: The _ was high.\nAnswer: the'` -> **cloud**
  - B: `'Of the cloud and the mud. Fill in the blank: The _ was low.\nAnswer: the'` -> **mud**
- order 2: `'Of the mud and the cloud. Fill in the blank: The _ was high.\nAnswer: the'` -> **cloud**
  - B: `'Of the mud and the cloud. Fill in the blank: The _ was low.\nAnswer: the'` -> **mud**

## 4. ctrl|stone|wool|hard|soft (objects, ext)
- order 1: `'Of the stone and the wool. Fill in the blank: The _ was hard.\nAnswer: the'` -> **stone**
  - B: `'Of the stone and the wool. Fill in the blank: The _ was soft.\nAnswer: the'` -> **wool**
- order 2: `'Of the wool and the stone. Fill in the blank: The _ was hard.\nAnswer: the'` -> **stone**
  - B: `'Of the wool and the stone. Fill in the blank: The _ was soft.\nAnswer: the'` -> **wool**

## 5. ctrl|song|funeral|happy|sad (objects, ext)
- order 1: `'Of the song and the funeral. Fill in the blank: The _ was happy.\nAnswer: the'` -> **song**
  - B: `'Of the song and the funeral. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**
- order 2: `'Of the funeral and the song. Fill in the blank: The _ was happy.\nAnswer: the'` -> **song**
  - B: `'Of the funeral and the song. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**

## 6. ctrl|vacation|commute|fun|boring (objects, ext)
- order 1: `'Of the vacation and the commute. Fill in the blank: The _ was fun.\nAnswer: the'` -> **vacation**
  - B: `'Of the vacation and the commute. Fill in the blank: The _ was boring.\nAnswer: the'` -> **commute**
- order 2: `'Of the commute and the vacation. Fill in the blank: The _ was fun.\nAnswer: the'` -> **vacation**
  - B: `'Of the commute and the vacation. Fill in the blank: The _ was boring.\nAnswer: the'` -> **commute**

## 7. ctrl|stove|freezer|hot|cold (objects, ext)
- order 1: `'Of the stove and the freezer. Fill in the blank: The _ was hot.\nAnswer: the'` -> **stove**
  - B: `'Of the stove and the freezer. Fill in the blank: The _ was cold.\nAnswer: the'` -> **freezer**
- order 2: `'Of the freezer and the stove. Fill in the blank: The _ was hot.\nAnswer: the'` -> **stove**
  - B: `'Of the freezer and the stove. Fill in the blank: The _ was cold.\nAnswer: the'` -> **freezer**

## 8. ctrl|salsa|yogurt|spicy|mild (objects, ext)
- order 1: `'Of the salsa and the yogurt. Fill in the blank: The _ was spicy.\nAnswer: the'` -> **salsa**
  - B: `'Of the salsa and the yogurt. Fill in the blank: The _ was mild.\nAnswer: the'` -> **yogurt**
- order 2: `'Of the yogurt and the salsa. Fill in the blank: The _ was spicy.\nAnswer: the'` -> **salsa**
  - B: `'Of the yogurt and the salsa. Fill in the blank: The _ was mild.\nAnswer: the'` -> **yogurt**

## 9. ctrl|concert|library|loud|quiet (objects, ext)
- order 1: `'Of the concert and the library. Fill in the blank: The _ was loud.\nAnswer: the'` -> **concert**
  - B: `'Of the concert and the library. Fill in the blank: The _ was quiet.\nAnswer: the'` -> **library**
- order 2: `'Of the library and the concert. Fill in the blank: The _ was loud.\nAnswer: the'` -> **concert**
  - B: `'Of the library and the concert. Fill in the blank: The _ was quiet.\nAnswer: the'` -> **library**

## 10. ctrl|giant|child|tall|short (objects, ext)
- order 1: `'Of the giant and the child. Fill in the blank: The _ was tall.\nAnswer: the'` -> **giant**
  - B: `'Of the giant and the child. Fill in the blank: The _ was short.\nAnswer: the'` -> **child**
- order 2: `'Of the child and the giant. Fill in the blank: The _ was tall.\nAnswer: the'` -> **giant**
  - B: `'Of the child and the giant. Fill in the blank: The _ was short.\nAnswer: the'` -> **child**

## 11. ctrl|hospital|dump|clean|dirty (objects, ext)
- order 1: `'Of the hospital and the dump. Fill in the blank: The _ was clean.\nAnswer: the'` -> **hospital**
  - B: `'Of the hospital and the dump. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **dump**
- order 2: `'Of the dump and the hospital. Fill in the blank: The _ was clean.\nAnswer: the'` -> **hospital**
  - B: `'Of the dump and the hospital. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **dump**

## 12. ctrl|candy|lemon|sweet|sour (objects, ext)
- order 1: `'Of the candy and the lemon. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **candy**
  - B: `'Of the candy and the lemon. Fill in the blank: The _ was sour.\nAnswer: the'` -> **lemon**
- order 2: `'Of the lemon and the candy. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **candy**
  - B: `'Of the lemon and the candy. Fill in the blank: The _ was sour.\nAnswer: the'` -> **lemon**

## 13. ctrl|lake|desert|wet|dry (objects, ext)
- order 1: `'Of the lake and the desert. Fill in the blank: The _ was wet.\nAnswer: the'` -> **lake**
  - B: `'Of the lake and the desert. Fill in the blank: The _ was dry.\nAnswer: the'` -> **desert**
- order 2: `'Of the desert and the lake. Fill in the blank: The _ was wet.\nAnswer: the'` -> **lake**
  - B: `'Of the desert and the lake. Fill in the blank: The _ was dry.\nAnswer: the'` -> **desert**

## 14. ctrl|knife|spoon|sharp|dull (objects, ext)
- order 1: `'Of the knife and the spoon. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **knife**
  - B: `'Of the knife and the spoon. Fill in the blank: The _ was dull.\nAnswer: the'` -> **spoon**
- order 2: `'Of the spoon and the knife. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **knife**
  - B: `'Of the spoon and the knife. Fill in the blank: The _ was dull.\nAnswer: the'` -> **spoon**

## 15. ctrl|soup|ice|hot|cold (objects, ext)
- order 1: `'Of the soup and the ice. Fill in the blank: The _ was hot.\nAnswer: the'` -> **soup**
  - B: `'Of the soup and the ice. Fill in the blank: The _ was cold.\nAnswer: the'` -> **ice**
- order 2: `'Of the ice and the soup. Fill in the blank: The _ was hot.\nAnswer: the'` -> **soup**
  - B: `'Of the ice and the soup. Fill in the blank: The _ was cold.\nAnswer: the'` -> **ice**

## 16. ctrl|bear|rabbit|big|small (objects, ext)
- order 1: `'Of the bear and the rabbit. Fill in the blank: The _ was big.\nAnswer: the'` -> **bear**
  - B: `'Of the bear and the rabbit. Fill in the blank: The _ was small.\nAnswer: the'` -> **rabbit**
- order 2: `'Of the rabbit and the bear. Fill in the blank: The _ was big.\nAnswer: the'` -> **bear**
  - B: `'Of the rabbit and the bear. Fill in the blank: The _ was small.\nAnswer: the'` -> **rabbit**

## 17. ctrl|egg|rock|fragile|sturdy (objects, ext)
- order 1: `'Of the egg and the rock. Fill in the blank: The _ was fragile.\nAnswer: the'` -> **egg**
  - B: `'Of the egg and the rock. Fill in the blank: The _ was sturdy.\nAnswer: the'` -> **rock**
- order 2: `'Of the rock and the egg. Fill in the blank: The _ was fragile.\nAnswer: the'` -> **egg**
  - B: `'Of the rock and the egg. Fill in the blank: The _ was sturdy.\nAnswer: the'` -> **rock**

## 18. ctrl|party|lecture|fun|boring (objects, ext)
- order 1: `'Of the party and the lecture. Fill in the blank: The _ was fun.\nAnswer: the'` -> **party**
  - B: `'Of the party and the lecture. Fill in the blank: The _ was boring.\nAnswer: the'` -> **lecture**
- order 2: `'Of the lecture and the party. Fill in the blank: The _ was fun.\nAnswer: the'` -> **party**
  - B: `'Of the lecture and the party. Fill in the blank: The _ was boring.\nAnswer: the'` -> **lecture**

## 19. ctrl|soap|dirt|clean|dirty (objects, ext)
- order 1: `'Of the soap and the dirt. Fill in the blank: The _ was clean.\nAnswer: the'` -> **soap**
  - B: `'Of the soap and the dirt. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **dirt**
- order 2: `'Of the dirt and the soap. Fill in the blank: The _ was clean.\nAnswer: the'` -> **soap**
  - B: `'Of the dirt and the soap. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **dirt**

## 20. ctrl|sword|stick|sharp|blunt (objects, ext)
- order 1: `'Of the sword and the stick. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **sword**
  - B: `'Of the sword and the stick. Fill in the blank: The _ was blunt.\nAnswer: the'` -> **stick**
- order 2: `'Of the stick and the sword. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **sword**
  - B: `'Of the stick and the sword. Fill in the blank: The _ was blunt.\nAnswer: the'` -> **stick**

## 21. ctrl|kale|cookie|bitter|sweet (objects, ext)
- order 1: `'Of the kale and the cookie. Fill in the blank: The _ was bitter.\nAnswer: the'` -> **kale**
  - B: `'Of the kale and the cookie. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **cookie**
- order 2: `'Of the cookie and the kale. Fill in the blank: The _ was bitter.\nAnswer: the'` -> **kale**
  - B: `'Of the cookie and the kale. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **cookie**

## 22. ctrl|movie|exam|fun|boring (objects, ext)
- order 1: `'Of the movie and the exam. Fill in the blank: The _ was fun.\nAnswer: the'` -> **movie**
  - B: `'Of the movie and the exam. Fill in the blank: The _ was boring.\nAnswer: the'` -> **exam**
- order 2: `'Of the exam and the movie. Fill in the blank: The _ was fun.\nAnswer: the'` -> **movie**
  - B: `'Of the exam and the movie. Fill in the blank: The _ was boring.\nAnswer: the'` -> **exam**

## 23. ctrl|river|path|wide|narrow (objects, ext)
- order 1: `'Of the river and the path. Fill in the blank: The _ was wide.\nAnswer: the'` -> **river**
  - B: `'Of the river and the path. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **path**
- order 2: `'Of the path and the river. Fill in the blank: The _ was wide.\nAnswer: the'` -> **river**
  - B: `'Of the path and the river. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **path**

## 24. 3O2Y2UIUCQSZRZPVQ54P4YAQ7QOFKP (objects, ext)
- order 1: `'Of the bedroom and the house. Fill in the blank: The _ was too large.\nAnswer: the'` -> **bedroom**
  - B: `'Of the bedroom and the house. Fill in the blank: The _ was too small.\nAnswer: the'` -> **house**
- order 2: `'Of the house and the bedroom. Fill in the blank: The _ was too large.\nAnswer: the'` -> **bedroom**
  - B: `'Of the house and the bedroom. Fill in the blank: The _ was too small.\nAnswer: the'` -> **house**

## 25. ctrl|carnival|meeting|fun|boring (objects, ext)
- order 1: `'Of the carnival and the meeting. Fill in the blank: The _ was fun.\nAnswer: the'` -> **carnival**
  - B: `'Of the carnival and the meeting. Fill in the blank: The _ was boring.\nAnswer: the'` -> **meeting**
- order 2: `'Of the meeting and the carnival. Fill in the blank: The _ was fun.\nAnswer: the'` -> **carnival**
  - B: `'Of the meeting and the carnival. Fill in the blank: The _ was boring.\nAnswer: the'` -> **meeting**

## 26. ctrl|tea|snow|hot|cold (objects, ext)
- order 1: `'Of the tea and the snow. Fill in the blank: The _ was hot.\nAnswer: the'` -> **tea**
  - B: `'Of the tea and the snow. Fill in the blank: The _ was cold.\nAnswer: the'` -> **snow**
- order 2: `'Of the snow and the tea. Fill in the blank: The _ was hot.\nAnswer: the'` -> **tea**
  - B: `'Of the snow and the tea. Fill in the blank: The _ was cold.\nAnswer: the'` -> **snow**

## 27. ctrl|road|path|long|short (objects, ext)
- order 1: `'Of the road and the path. Fill in the blank: The _ was long.\nAnswer: the'` -> **road**
  - B: `'Of the road and the path. Fill in the blank: The _ was short.\nAnswer: the'` -> **path**
- order 2: `'Of the path and the road. Fill in the blank: The _ was long.\nAnswer: the'` -> **road**
  - B: `'Of the path and the road. Fill in the blank: The _ was short.\nAnswer: the'` -> **path**

## 28. ctrl|snow|mud|clean|dirty (objects, ext)
- order 1: `'Of the snow and the mud. Fill in the blank: The _ was clean.\nAnswer: the'` -> **snow**
  - B: `'Of the snow and the mud. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **mud**
- order 2: `'Of the mud and the snow. Fill in the blank: The _ was clean.\nAnswer: the'` -> **snow**
  - B: `'Of the mud and the snow. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **mud**

## 29. 38DCH97KHJGG1T8T6YR77G20O6RQJ5 (objects, ext)
- order 1: `'Of the cake and the card. Fill in the blank: The _ was delicious.\nAnswer: the'` -> **cake**
  - B: `'Of the cake and the card. Fill in the blank: The _ was sentimental.\nAnswer: the'` -> **card**
- order 2: `'Of the card and the cake. Fill in the blank: The _ was delicious.\nAnswer: the'` -> **cake**
  - B: `'Of the card and the cake. Fill in the blank: The _ was sentimental.\nAnswer: the'` -> **card**

## 30. ctrl|planet|marble|big|small (objects, ext)
- order 1: `'Of the planet and the marble. Fill in the blank: The _ was big.\nAnswer: the'` -> **planet**
  - B: `'Of the planet and the marble. Fill in the blank: The _ was small.\nAnswer: the'` -> **marble**
- order 2: `'Of the marble and the planet. Fill in the blank: The _ was big.\nAnswer: the'` -> **planet**
  - B: `'Of the marble and the planet. Fill in the blank: The _ was small.\nAnswer: the'` -> **marble**

## 31. ctrl|rock|feather|heavy|light (objects, ext)
- order 1: `'Of the rock and the feather. Fill in the blank: The _ was heavy.\nAnswer: the'` -> **rock**
  - B: `'Of the rock and the feather. Fill in the blank: The _ was light.\nAnswer: the'` -> **feather**
- order 2: `'Of the feather and the rock. Fill in the blank: The _ was heavy.\nAnswer: the'` -> **rock**
  - B: `'Of the feather and the rock. Fill in the blank: The _ was light.\nAnswer: the'` -> **feather**

## 32. ctrl|cow|rat|big|small (objects, ext)
- order 1: `'Of the cow and the rat. Fill in the blank: The _ was big.\nAnswer: the'` -> **cow**
  - B: `'Of the cow and the rat. Fill in the blank: The _ was small.\nAnswer: the'` -> **rat**
- order 2: `'Of the rat and the cow. Fill in the blank: The _ was big.\nAnswer: the'` -> **cow**
  - B: `'Of the rat and the cow. Fill in the blank: The _ was small.\nAnswer: the'` -> **rat**

## 33. ctrl|jet|boat|fast|slow (objects, ext)
- order 1: `'Of the jet and the boat. Fill in the blank: The _ was fast.\nAnswer: the'` -> **jet**
  - B: `'Of the jet and the boat. Fill in the blank: The _ was slow.\nAnswer: the'` -> **boat**
- order 2: `'Of the boat and the jet. Fill in the blank: The _ was fast.\nAnswer: the'` -> **jet**
  - B: `'Of the boat and the jet. Fill in the blank: The _ was slow.\nAnswer: the'` -> **boat**

## 34. ctrl|wolf|sheep|dangerous|safe (objects, ext)
- order 1: `'Of the wolf and the sheep. Fill in the blank: The _ was dangerous.\nAnswer: the'` -> **wolf**
  - B: `'Of the wolf and the sheep. Fill in the blank: The _ was safe.\nAnswer: the'` -> **sheep**
- order 2: `'Of the sheep and the wolf. Fill in the blank: The _ was dangerous.\nAnswer: the'` -> **wolf**
  - B: `'Of the sheep and the wolf. Fill in the blank: The _ was safe.\nAnswer: the'` -> **sheep**

## 35. ctrl|tea|ice|warm|cold (objects, ext)
- order 1: `'Of the tea and the ice. Fill in the blank: The _ was warm.\nAnswer: the'` -> **tea**
  - B: `'Of the tea and the ice. Fill in the blank: The _ was cold.\nAnswer: the'` -> **ice**
- order 2: `'Of the ice and the tea. Fill in the blank: The _ was warm.\nAnswer: the'` -> **tea**
  - B: `'Of the ice and the tea. Fill in the blank: The _ was cold.\nAnswer: the'` -> **ice**

## 36. ctrl|steel|cotton|hard|soft (objects, ext)
- order 1: `'Of the steel and the cotton. Fill in the blank: The _ was hard.\nAnswer: the'` -> **steel**
  - B: `'Of the steel and the cotton. Fill in the blank: The _ was soft.\nAnswer: the'` -> **cotton**
- order 2: `'Of the cotton and the steel. Fill in the blank: The _ was hard.\nAnswer: the'` -> **steel**
  - B: `'Of the cotton and the steel. Fill in the blank: The _ was soft.\nAnswer: the'` -> **cotton**

## 37. ctrl|palace|cabin|big|small (objects, ext)
- order 1: `'Of the palace and the cabin. Fill in the blank: The _ was big.\nAnswer: the'` -> **palace**
  - B: `'Of the palace and the cabin. Fill in the blank: The _ was small.\nAnswer: the'` -> **cabin**
- order 2: `'Of the cabin and the palace. Fill in the blank: The _ was big.\nAnswer: the'` -> **palace**
  - B: `'Of the cabin and the palace. Fill in the blank: The _ was small.\nAnswer: the'` -> **cabin**

## 38. ctrl|car|bike|fast|slow (objects, ext)
- order 1: `'Of the car and the bike. Fill in the blank: The _ was fast.\nAnswer: the'` -> **car**
  - B: `'Of the car and the bike. Fill in the blank: The _ was slow.\nAnswer: the'` -> **bike**
- order 2: `'Of the bike and the car. Fill in the blank: The _ was fast.\nAnswer: the'` -> **car**
  - B: `'Of the bike and the car. Fill in the blank: The _ was slow.\nAnswer: the'` -> **bike**

## 39. ctrl|ocean|stream|wide|narrow (objects, ext)
- order 1: `'Of the ocean and the stream. Fill in the blank: The _ was wide.\nAnswer: the'` -> **ocean**
  - B: `'Of the ocean and the stream. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **stream**
- order 2: `'Of the stream and the ocean. Fill in the blank: The _ was wide.\nAnswer: the'` -> **ocean**
  - B: `'Of the stream and the ocean. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **stream**

## 40. ctrl|castle|hut|big|small (objects, ext)
- order 1: `'Of the castle and the hut. Fill in the blank: The _ was big.\nAnswer: the'` -> **castle**
  - B: `'Of the castle and the hut. Fill in the blank: The _ was small.\nAnswer: the'` -> **hut**
- order 2: `'Of the hut and the castle. Fill in the blank: The _ was big.\nAnswer: the'` -> **castle**
  - B: `'Of the hut and the castle. Fill in the blank: The _ was small.\nAnswer: the'` -> **hut**

## 41. ctrl|iron|velvet|hard|soft (objects, ext)
- order 1: `'Of the iron and the velvet. Fill in the blank: The _ was hard.\nAnswer: the'` -> **iron**
  - B: `'Of the iron and the velvet. Fill in the blank: The _ was soft.\nAnswer: the'` -> **velvet**
- order 2: `'Of the velvet and the iron. Fill in the blank: The _ was hard.\nAnswer: the'` -> **iron**
  - B: `'Of the velvet and the iron. Fill in the blank: The _ was soft.\nAnswer: the'` -> **velvet**

## 42. ctrl|lake|dust|wet|dry (objects, ext)
- order 1: `'Of the lake and the dust. Fill in the blank: The _ was wet.\nAnswer: the'` -> **lake**
  - B: `'Of the lake and the dust. Fill in the blank: The _ was dry.\nAnswer: the'` -> **dust**
- order 2: `'Of the dust and the lake. Fill in the blank: The _ was wet.\nAnswer: the'` -> **lake**
  - B: `'Of the dust and the lake. Fill in the blank: The _ was dry.\nAnswer: the'` -> **dust**

## 43. ctrl|horse|ant|big|small (objects, ext)
- order 1: `'Of the horse and the ant. Fill in the blank: The _ was big.\nAnswer: the'` -> **horse**
  - B: `'Of the horse and the ant. Fill in the blank: The _ was small.\nAnswer: the'` -> **ant**
- order 2: `'Of the ant and the horse. Fill in the blank: The _ was big.\nAnswer: the'` -> **horse**
  - B: `'Of the ant and the horse. Fill in the blank: The _ was small.\nAnswer: the'` -> **ant**

## 44. ctrl|ocean|desert|wet|dry (objects, ext)
- order 1: `'Of the ocean and the desert. Fill in the blank: The _ was wet.\nAnswer: the'` -> **ocean**
  - B: `'Of the ocean and the desert. Fill in the blank: The _ was dry.\nAnswer: the'` -> **desert**
- order 2: `'Of the desert and the ocean. Fill in the blank: The _ was wet.\nAnswer: the'` -> **ocean**
  - B: `'Of the desert and the ocean. Fill in the blank: The _ was dry.\nAnswer: the'` -> **desert**

## 45. ctrl|stadium|graveyard|full|empty (objects, ext)
- order 1: `'Of the stadium and the graveyard. Fill in the blank: The _ was full.\nAnswer: the'` -> **stadium**
  - B: `'Of the stadium and the graveyard. Fill in the blank: The _ was empty.\nAnswer: the'` -> **graveyard**
- order 2: `'Of the graveyard and the stadium. Fill in the blank: The _ was full.\nAnswer: the'` -> **stadium**
  - B: `'Of the graveyard and the stadium. Fill in the blank: The _ was empty.\nAnswer: the'` -> **graveyard**

## 46. ctrl|road|path|wide|narrow (objects, ext)
- order 1: `'Of the road and the path. Fill in the blank: The _ was wide.\nAnswer: the'` -> **road**
  - B: `'Of the road and the path. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **path**
- order 2: `'Of the path and the road. Fill in the blank: The _ was wide.\nAnswer: the'` -> **road**
  - B: `'Of the path and the road. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **path**

## 47. ctrl|train|wagon|fast|slow (objects, ext)
- order 1: `'Of the train and the wagon. Fill in the blank: The _ was fast.\nAnswer: the'` -> **train**
  - B: `'Of the train and the wagon. Fill in the blank: The _ was slow.\nAnswer: the'` -> **wagon**
- order 2: `'Of the wagon and the train. Fill in the blank: The _ was fast.\nAnswer: the'` -> **train**
  - B: `'Of the wagon and the train. Fill in the blank: The _ was slow.\nAnswer: the'` -> **wagon**

## 48. 3HYV4299H2A2K9M563ZGO948PS98ER (objects, core)
- order 1: `'Of the head and the crowd. Fill in the blank: The _ was bald.\nAnswer: the'` -> **head**
  - B: `'Of the head and the crowd. Fill in the blank: The _ was vast.\nAnswer: the'` -> **crowd**
- order 2: `'Of the crowd and the head. Fill in the blank: The _ was bald.\nAnswer: the'` -> **head**
  - B: `'Of the crowd and the head. Fill in the blank: The _ was vast.\nAnswer: the'` -> **crowd**

## 49. ctrl|airplane|bee|large|tiny (objects, ext)
- order 1: `'Of the airplane and the bee. Fill in the blank: The _ was large.\nAnswer: the'` -> **airplane**
  - B: `'Of the airplane and the bee. Fill in the blank: The _ was tiny.\nAnswer: the'` -> **bee**
- order 2: `'Of the bee and the airplane. Fill in the blank: The _ was large.\nAnswer: the'` -> **airplane**
  - B: `'Of the bee and the airplane. Fill in the blank: The _ was tiny.\nAnswer: the'` -> **bee**

## 50. ctrl|highway|graveyard|noisy|silent (objects, ext)
- order 1: `'Of the highway and the graveyard. Fill in the blank: The _ was noisy.\nAnswer: the'` -> **highway**
  - B: `'Of the highway and the graveyard. Fill in the blank: The _ was silent.\nAnswer: the'` -> **graveyard**
- order 2: `'Of the graveyard and the highway. Fill in the blank: The _ was noisy.\nAnswer: the'` -> **highway**
  - B: `'Of the graveyard and the highway. Fill in the blank: The _ was silent.\nAnswer: the'` -> **graveyard**
