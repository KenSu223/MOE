# dA_e1_a2e: 50 seeded samples (qwen-family texts; random_state 0)

Each item: prompt A -> answer A | prompt B -> answer B (STR at token positions str_pos). Orders shown where built.

## 1. ctrl|party|funeral|happy|sad (objects, ext)
- order 1: `'The party was happy and the funeral was sad. Fill in the blank: The _ was happy.\nAnswer: the'` -> **party**
  - B: `'The party was happy and the funeral was sad. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**
- order 1: `'The party was sad and the funeral was happy. Fill in the blank: The _ was happy.\nAnswer: the'` -> **funeral**
  - B: `'The party was sad and the funeral was happy. Fill in the blank: The _ was sad.\nAnswer: the'` -> **party**
- order 2: `'The funeral was sad and the party was happy. Fill in the blank: The _ was happy.\nAnswer: the'` -> **party**
  - B: `'The funeral was sad and the party was happy. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**
- order 2: `'The funeral was happy and the party was sad. Fill in the blank: The _ was happy.\nAnswer: the'` -> **funeral**
  - B: `'The funeral was happy and the party was sad. Fill in the blank: The _ was sad.\nAnswer: the'` -> **party**

## 2. ctrl|birthday|funeral|happy|sad (objects, ext)
- order 1: `'The birthday was happy and the funeral was sad. Fill in the blank: The _ was happy.\nAnswer: the'` -> **birthday**
  - B: `'The birthday was happy and the funeral was sad. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**
- order 1: `'The birthday was sad and the funeral was happy. Fill in the blank: The _ was happy.\nAnswer: the'` -> **funeral**
  - B: `'The birthday was sad and the funeral was happy. Fill in the blank: The _ was sad.\nAnswer: the'` -> **birthday**
- order 2: `'The funeral was sad and the birthday was happy. Fill in the blank: The _ was happy.\nAnswer: the'` -> **birthday**
  - B: `'The funeral was sad and the birthday was happy. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**
- order 2: `'The funeral was happy and the birthday was sad. Fill in the blank: The _ was happy.\nAnswer: the'` -> **funeral**
  - B: `'The funeral was happy and the birthday was sad. Fill in the blank: The _ was sad.\nAnswer: the'` -> **birthday**

## 3. ctrl|cloud|mud|high|low (objects, ext)
- order 1: `'The cloud was high and the mud was low. Fill in the blank: The _ was high.\nAnswer: the'` -> **cloud**
  - B: `'The cloud was high and the mud was low. Fill in the blank: The _ was low.\nAnswer: the'` -> **mud**
- order 1: `'The cloud was low and the mud was high. Fill in the blank: The _ was high.\nAnswer: the'` -> **mud**
  - B: `'The cloud was low and the mud was high. Fill in the blank: The _ was low.\nAnswer: the'` -> **cloud**
- order 2: `'The mud was low and the cloud was high. Fill in the blank: The _ was high.\nAnswer: the'` -> **cloud**
  - B: `'The mud was low and the cloud was high. Fill in the blank: The _ was low.\nAnswer: the'` -> **mud**
- order 2: `'The mud was high and the cloud was low. Fill in the blank: The _ was high.\nAnswer: the'` -> **mud**
  - B: `'The mud was high and the cloud was low. Fill in the blank: The _ was low.\nAnswer: the'` -> **cloud**

## 4. ctrl|stone|wool|hard|soft (objects, ext)
- order 1: `'The stone was hard and the wool was soft. Fill in the blank: The _ was hard.\nAnswer: the'` -> **stone**
  - B: `'The stone was hard and the wool was soft. Fill in the blank: The _ was soft.\nAnswer: the'` -> **wool**
- order 1: `'The stone was soft and the wool was hard. Fill in the blank: The _ was hard.\nAnswer: the'` -> **wool**
  - B: `'The stone was soft and the wool was hard. Fill in the blank: The _ was soft.\nAnswer: the'` -> **stone**
- order 2: `'The wool was soft and the stone was hard. Fill in the blank: The _ was hard.\nAnswer: the'` -> **stone**
  - B: `'The wool was soft and the stone was hard. Fill in the blank: The _ was soft.\nAnswer: the'` -> **wool**
- order 2: `'The wool was hard and the stone was soft. Fill in the blank: The _ was hard.\nAnswer: the'` -> **wool**
  - B: `'The wool was hard and the stone was soft. Fill in the blank: The _ was soft.\nAnswer: the'` -> **stone**

## 5. ctrl|song|funeral|happy|sad (objects, ext)
- order 1: `'The song was happy and the funeral was sad. Fill in the blank: The _ was happy.\nAnswer: the'` -> **song**
  - B: `'The song was happy and the funeral was sad. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**
- order 1: `'The song was sad and the funeral was happy. Fill in the blank: The _ was happy.\nAnswer: the'` -> **funeral**
  - B: `'The song was sad and the funeral was happy. Fill in the blank: The _ was sad.\nAnswer: the'` -> **song**
- order 2: `'The funeral was sad and the song was happy. Fill in the blank: The _ was happy.\nAnswer: the'` -> **song**
  - B: `'The funeral was sad and the song was happy. Fill in the blank: The _ was sad.\nAnswer: the'` -> **funeral**
- order 2: `'The funeral was happy and the song was sad. Fill in the blank: The _ was happy.\nAnswer: the'` -> **funeral**
  - B: `'The funeral was happy and the song was sad. Fill in the blank: The _ was sad.\nAnswer: the'` -> **song**

## 6. ctrl|vacation|commute|fun|boring (objects, ext)
- order 1: `'The vacation was fun and the commute was boring. Fill in the blank: The _ was fun.\nAnswer: the'` -> **vacation**
  - B: `'The vacation was fun and the commute was boring. Fill in the blank: The _ was boring.\nAnswer: the'` -> **commute**
- order 1: `'The vacation was boring and the commute was fun. Fill in the blank: The _ was fun.\nAnswer: the'` -> **commute**
  - B: `'The vacation was boring and the commute was fun. Fill in the blank: The _ was boring.\nAnswer: the'` -> **vacation**
- order 2: `'The commute was boring and the vacation was fun. Fill in the blank: The _ was fun.\nAnswer: the'` -> **vacation**
  - B: `'The commute was boring and the vacation was fun. Fill in the blank: The _ was boring.\nAnswer: the'` -> **commute**
- order 2: `'The commute was fun and the vacation was boring. Fill in the blank: The _ was fun.\nAnswer: the'` -> **commute**
  - B: `'The commute was fun and the vacation was boring. Fill in the blank: The _ was boring.\nAnswer: the'` -> **vacation**

## 7. ctrl|stove|freezer|hot|cold (objects, ext)
- order 1: `'The stove was hot and the freezer was cold. Fill in the blank: The _ was hot.\nAnswer: the'` -> **stove**
  - B: `'The stove was hot and the freezer was cold. Fill in the blank: The _ was cold.\nAnswer: the'` -> **freezer**
- order 1: `'The stove was cold and the freezer was hot. Fill in the blank: The _ was hot.\nAnswer: the'` -> **freezer**
  - B: `'The stove was cold and the freezer was hot. Fill in the blank: The _ was cold.\nAnswer: the'` -> **stove**
- order 2: `'The freezer was cold and the stove was hot. Fill in the blank: The _ was hot.\nAnswer: the'` -> **stove**
  - B: `'The freezer was cold and the stove was hot. Fill in the blank: The _ was cold.\nAnswer: the'` -> **freezer**
- order 2: `'The freezer was hot and the stove was cold. Fill in the blank: The _ was hot.\nAnswer: the'` -> **freezer**
  - B: `'The freezer was hot and the stove was cold. Fill in the blank: The _ was cold.\nAnswer: the'` -> **stove**

## 8. ctrl|salsa|yogurt|spicy|mild (objects, ext)
- order 1: `'The salsa was spicy and the yogurt was mild. Fill in the blank: The _ was spicy.\nAnswer: the'` -> **salsa**
  - B: `'The salsa was spicy and the yogurt was mild. Fill in the blank: The _ was mild.\nAnswer: the'` -> **yogurt**
- order 1: `'The salsa was mild and the yogurt was spicy. Fill in the blank: The _ was spicy.\nAnswer: the'` -> **yogurt**
  - B: `'The salsa was mild and the yogurt was spicy. Fill in the blank: The _ was mild.\nAnswer: the'` -> **salsa**
- order 2: `'The yogurt was mild and the salsa was spicy. Fill in the blank: The _ was spicy.\nAnswer: the'` -> **salsa**
  - B: `'The yogurt was mild and the salsa was spicy. Fill in the blank: The _ was mild.\nAnswer: the'` -> **yogurt**
- order 2: `'The yogurt was spicy and the salsa was mild. Fill in the blank: The _ was spicy.\nAnswer: the'` -> **yogurt**
  - B: `'The yogurt was spicy and the salsa was mild. Fill in the blank: The _ was mild.\nAnswer: the'` -> **salsa**

## 9. ctrl|concert|library|loud|quiet (objects, ext)
- order 1: `'The concert was loud and the library was quiet. Fill in the blank: The _ was loud.\nAnswer: the'` -> **concert**
  - B: `'The concert was loud and the library was quiet. Fill in the blank: The _ was quiet.\nAnswer: the'` -> **library**
- order 1: `'The concert was quiet and the library was loud. Fill in the blank: The _ was loud.\nAnswer: the'` -> **library**
  - B: `'The concert was quiet and the library was loud. Fill in the blank: The _ was quiet.\nAnswer: the'` -> **concert**
- order 2: `'The library was quiet and the concert was loud. Fill in the blank: The _ was loud.\nAnswer: the'` -> **concert**
  - B: `'The library was quiet and the concert was loud. Fill in the blank: The _ was quiet.\nAnswer: the'` -> **library**
- order 2: `'The library was loud and the concert was quiet. Fill in the blank: The _ was loud.\nAnswer: the'` -> **library**
  - B: `'The library was loud and the concert was quiet. Fill in the blank: The _ was quiet.\nAnswer: the'` -> **concert**

## 10. ctrl|giant|child|tall|short (objects, ext)
- order 1: `'The giant was tall and the child was short. Fill in the blank: The _ was tall.\nAnswer: the'` -> **giant**
  - B: `'The giant was tall and the child was short. Fill in the blank: The _ was short.\nAnswer: the'` -> **child**
- order 1: `'The giant was short and the child was tall. Fill in the blank: The _ was tall.\nAnswer: the'` -> **child**
  - B: `'The giant was short and the child was tall. Fill in the blank: The _ was short.\nAnswer: the'` -> **giant**
- order 2: `'The child was short and the giant was tall. Fill in the blank: The _ was tall.\nAnswer: the'` -> **giant**
  - B: `'The child was short and the giant was tall. Fill in the blank: The _ was short.\nAnswer: the'` -> **child**
- order 2: `'The child was tall and the giant was short. Fill in the blank: The _ was tall.\nAnswer: the'` -> **child**
  - B: `'The child was tall and the giant was short. Fill in the blank: The _ was short.\nAnswer: the'` -> **giant**

## 11. ctrl|hospital|dump|clean|dirty (objects, ext)
- order 1: `'The hospital was clean and the dump was dirty. Fill in the blank: The _ was clean.\nAnswer: the'` -> **hospital**
  - B: `'The hospital was clean and the dump was dirty. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **dump**
- order 1: `'The hospital was dirty and the dump was clean. Fill in the blank: The _ was clean.\nAnswer: the'` -> **dump**
  - B: `'The hospital was dirty and the dump was clean. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **hospital**
- order 2: `'The dump was dirty and the hospital was clean. Fill in the blank: The _ was clean.\nAnswer: the'` -> **hospital**
  - B: `'The dump was dirty and the hospital was clean. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **dump**
- order 2: `'The dump was clean and the hospital was dirty. Fill in the blank: The _ was clean.\nAnswer: the'` -> **dump**
  - B: `'The dump was clean and the hospital was dirty. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **hospital**

## 12. ctrl|candy|lemon|sweet|sour (objects, ext)
- order 1: `'The candy was sweet and the lemon was sour. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **candy**
  - B: `'The candy was sweet and the lemon was sour. Fill in the blank: The _ was sour.\nAnswer: the'` -> **lemon**
- order 1: `'The candy was sour and the lemon was sweet. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **lemon**
  - B: `'The candy was sour and the lemon was sweet. Fill in the blank: The _ was sour.\nAnswer: the'` -> **candy**
- order 2: `'The lemon was sour and the candy was sweet. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **candy**
  - B: `'The lemon was sour and the candy was sweet. Fill in the blank: The _ was sour.\nAnswer: the'` -> **lemon**
- order 2: `'The lemon was sweet and the candy was sour. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **lemon**
  - B: `'The lemon was sweet and the candy was sour. Fill in the blank: The _ was sour.\nAnswer: the'` -> **candy**

## 13. ctrl|lake|desert|wet|dry (objects, ext)
- order 1: `'The lake was wet and the desert was dry. Fill in the blank: The _ was wet.\nAnswer: the'` -> **lake**
  - B: `'The lake was wet and the desert was dry. Fill in the blank: The _ was dry.\nAnswer: the'` -> **desert**
- order 1: `'The lake was dry and the desert was wet. Fill in the blank: The _ was wet.\nAnswer: the'` -> **desert**
  - B: `'The lake was dry and the desert was wet. Fill in the blank: The _ was dry.\nAnswer: the'` -> **lake**
- order 2: `'The desert was dry and the lake was wet. Fill in the blank: The _ was wet.\nAnswer: the'` -> **lake**
  - B: `'The desert was dry and the lake was wet. Fill in the blank: The _ was dry.\nAnswer: the'` -> **desert**
- order 2: `'The desert was wet and the lake was dry. Fill in the blank: The _ was wet.\nAnswer: the'` -> **desert**
  - B: `'The desert was wet and the lake was dry. Fill in the blank: The _ was dry.\nAnswer: the'` -> **lake**

## 14. ctrl|knife|spoon|sharp|dull (objects, ext)
- order 1: `'The knife was sharp and the spoon was dull. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **knife**
  - B: `'The knife was sharp and the spoon was dull. Fill in the blank: The _ was dull.\nAnswer: the'` -> **spoon**
- order 1: `'The knife was dull and the spoon was sharp. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **spoon**
  - B: `'The knife was dull and the spoon was sharp. Fill in the blank: The _ was dull.\nAnswer: the'` -> **knife**
- order 2: `'The spoon was dull and the knife was sharp. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **knife**
  - B: `'The spoon was dull and the knife was sharp. Fill in the blank: The _ was dull.\nAnswer: the'` -> **spoon**
- order 2: `'The spoon was sharp and the knife was dull. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **spoon**
  - B: `'The spoon was sharp and the knife was dull. Fill in the blank: The _ was dull.\nAnswer: the'` -> **knife**

## 15. ctrl|soup|ice|hot|cold (objects, ext)
- order 1: `'The soup was hot and the ice was cold. Fill in the blank: The _ was hot.\nAnswer: the'` -> **soup**
  - B: `'The soup was hot and the ice was cold. Fill in the blank: The _ was cold.\nAnswer: the'` -> **ice**
- order 1: `'The soup was cold and the ice was hot. Fill in the blank: The _ was hot.\nAnswer: the'` -> **ice**
  - B: `'The soup was cold and the ice was hot. Fill in the blank: The _ was cold.\nAnswer: the'` -> **soup**
- order 2: `'The ice was cold and the soup was hot. Fill in the blank: The _ was hot.\nAnswer: the'` -> **soup**
  - B: `'The ice was cold and the soup was hot. Fill in the blank: The _ was cold.\nAnswer: the'` -> **ice**
- order 2: `'The ice was hot and the soup was cold. Fill in the blank: The _ was hot.\nAnswer: the'` -> **ice**
  - B: `'The ice was hot and the soup was cold. Fill in the blank: The _ was cold.\nAnswer: the'` -> **soup**

## 16. ctrl|bear|rabbit|big|small (objects, ext)
- order 1: `'The bear was big and the rabbit was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **bear**
  - B: `'The bear was big and the rabbit was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **rabbit**
- order 1: `'The bear was small and the rabbit was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **rabbit**
  - B: `'The bear was small and the rabbit was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **bear**
- order 2: `'The rabbit was small and the bear was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **bear**
  - B: `'The rabbit was small and the bear was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **rabbit**
- order 2: `'The rabbit was big and the bear was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **rabbit**
  - B: `'The rabbit was big and the bear was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **bear**

## 17. ctrl|egg|rock|fragile|sturdy (objects, ext)
- order 1: `'The egg was fragile and the rock was sturdy. Fill in the blank: The _ was fragile.\nAnswer: the'` -> **egg**
  - B: `'The egg was fragile and the rock was sturdy. Fill in the blank: The _ was sturdy.\nAnswer: the'` -> **rock**
- order 1: `'The egg was sturdy and the rock was fragile. Fill in the blank: The _ was fragile.\nAnswer: the'` -> **rock**
  - B: `'The egg was sturdy and the rock was fragile. Fill in the blank: The _ was sturdy.\nAnswer: the'` -> **egg**
- order 2: `'The rock was sturdy and the egg was fragile. Fill in the blank: The _ was fragile.\nAnswer: the'` -> **egg**
  - B: `'The rock was sturdy and the egg was fragile. Fill in the blank: The _ was sturdy.\nAnswer: the'` -> **rock**
- order 2: `'The rock was fragile and the egg was sturdy. Fill in the blank: The _ was fragile.\nAnswer: the'` -> **rock**
  - B: `'The rock was fragile and the egg was sturdy. Fill in the blank: The _ was sturdy.\nAnswer: the'` -> **egg**

## 18. ctrl|party|lecture|fun|boring (objects, ext)
- order 1: `'The party was fun and the lecture was boring. Fill in the blank: The _ was fun.\nAnswer: the'` -> **party**
  - B: `'The party was fun and the lecture was boring. Fill in the blank: The _ was boring.\nAnswer: the'` -> **lecture**
- order 1: `'The party was boring and the lecture was fun. Fill in the blank: The _ was fun.\nAnswer: the'` -> **lecture**
  - B: `'The party was boring and the lecture was fun. Fill in the blank: The _ was boring.\nAnswer: the'` -> **party**
- order 2: `'The lecture was boring and the party was fun. Fill in the blank: The _ was fun.\nAnswer: the'` -> **party**
  - B: `'The lecture was boring and the party was fun. Fill in the blank: The _ was boring.\nAnswer: the'` -> **lecture**
- order 2: `'The lecture was fun and the party was boring. Fill in the blank: The _ was fun.\nAnswer: the'` -> **lecture**
  - B: `'The lecture was fun and the party was boring. Fill in the blank: The _ was boring.\nAnswer: the'` -> **party**

## 19. ctrl|soap|dirt|clean|dirty (objects, ext)
- order 1: `'The soap was clean and the dirt was dirty. Fill in the blank: The _ was clean.\nAnswer: the'` -> **soap**
  - B: `'The soap was clean and the dirt was dirty. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **dirt**
- order 1: `'The soap was dirty and the dirt was clean. Fill in the blank: The _ was clean.\nAnswer: the'` -> **dirt**
  - B: `'The soap was dirty and the dirt was clean. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **soap**
- order 2: `'The dirt was dirty and the soap was clean. Fill in the blank: The _ was clean.\nAnswer: the'` -> **soap**
  - B: `'The dirt was dirty and the soap was clean. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **dirt**
- order 2: `'The dirt was clean and the soap was dirty. Fill in the blank: The _ was clean.\nAnswer: the'` -> **dirt**
  - B: `'The dirt was clean and the soap was dirty. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **soap**

## 20. ctrl|sword|stick|sharp|blunt (objects, ext)
- order 1: `'The sword was sharp and the stick was blunt. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **sword**
  - B: `'The sword was sharp and the stick was blunt. Fill in the blank: The _ was blunt.\nAnswer: the'` -> **stick**
- order 1: `'The sword was blunt and the stick was sharp. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **stick**
  - B: `'The sword was blunt and the stick was sharp. Fill in the blank: The _ was blunt.\nAnswer: the'` -> **sword**
- order 2: `'The stick was blunt and the sword was sharp. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **sword**
  - B: `'The stick was blunt and the sword was sharp. Fill in the blank: The _ was blunt.\nAnswer: the'` -> **stick**
- order 2: `'The stick was sharp and the sword was blunt. Fill in the blank: The _ was sharp.\nAnswer: the'` -> **stick**
  - B: `'The stick was sharp and the sword was blunt. Fill in the blank: The _ was blunt.\nAnswer: the'` -> **sword**

## 21. ctrl|kale|cookie|bitter|sweet (objects, ext)
- order 1: `'The kale was bitter and the cookie was sweet. Fill in the blank: The _ was bitter.\nAnswer: the'` -> **kale**
  - B: `'The kale was bitter and the cookie was sweet. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **cookie**
- order 1: `'The kale was sweet and the cookie was bitter. Fill in the blank: The _ was bitter.\nAnswer: the'` -> **cookie**
  - B: `'The kale was sweet and the cookie was bitter. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **kale**
- order 2: `'The cookie was sweet and the kale was bitter. Fill in the blank: The _ was bitter.\nAnswer: the'` -> **kale**
  - B: `'The cookie was sweet and the kale was bitter. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **cookie**
- order 2: `'The cookie was bitter and the kale was sweet. Fill in the blank: The _ was bitter.\nAnswer: the'` -> **cookie**
  - B: `'The cookie was bitter and the kale was sweet. Fill in the blank: The _ was sweet.\nAnswer: the'` -> **kale**

## 22. ctrl|movie|exam|fun|boring (objects, ext)
- order 1: `'The movie was fun and the exam was boring. Fill in the blank: The _ was fun.\nAnswer: the'` -> **movie**
  - B: `'The movie was fun and the exam was boring. Fill in the blank: The _ was boring.\nAnswer: the'` -> **exam**
- order 1: `'The movie was boring and the exam was fun. Fill in the blank: The _ was fun.\nAnswer: the'` -> **exam**
  - B: `'The movie was boring and the exam was fun. Fill in the blank: The _ was boring.\nAnswer: the'` -> **movie**
- order 2: `'The exam was boring and the movie was fun. Fill in the blank: The _ was fun.\nAnswer: the'` -> **movie**
  - B: `'The exam was boring and the movie was fun. Fill in the blank: The _ was boring.\nAnswer: the'` -> **exam**
- order 2: `'The exam was fun and the movie was boring. Fill in the blank: The _ was fun.\nAnswer: the'` -> **exam**
  - B: `'The exam was fun and the movie was boring. Fill in the blank: The _ was boring.\nAnswer: the'` -> **movie**

## 23. ctrl|river|path|wide|narrow (objects, ext)
- order 1: `'The river was wide and the path was narrow. Fill in the blank: The _ was wide.\nAnswer: the'` -> **river**
  - B: `'The river was wide and the path was narrow. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **path**
- order 1: `'The river was narrow and the path was wide. Fill in the blank: The _ was wide.\nAnswer: the'` -> **path**
  - B: `'The river was narrow and the path was wide. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **river**
- order 2: `'The path was narrow and the river was wide. Fill in the blank: The _ was wide.\nAnswer: the'` -> **river**
  - B: `'The path was narrow and the river was wide. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **path**
- order 2: `'The path was wide and the river was narrow. Fill in the blank: The _ was wide.\nAnswer: the'` -> **path**
  - B: `'The path was wide and the river was narrow. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **river**

## 24. 3O2Y2UIUCQSZRZPVQ54P4YAQ7QOFKP (objects, ext)
- order 1: `'The bedroom was too large and the house was too small. Fill in the blank: The _ was too large.\nAnswer: the'` -> **bedroom**
  - B: `'The bedroom was too large and the house was too small. Fill in the blank: The _ was too small.\nAnswer: the'` -> **house**
- order 2: `'The house was too small and the bedroom was too large. Fill in the blank: The _ was too large.\nAnswer: the'` -> **bedroom**
  - B: `'The house was too small and the bedroom was too large. Fill in the blank: The _ was too small.\nAnswer: the'` -> **house**

## 25. ctrl|carnival|meeting|fun|boring (objects, ext)
- order 1: `'The carnival was fun and the meeting was boring. Fill in the blank: The _ was fun.\nAnswer: the'` -> **carnival**
  - B: `'The carnival was fun and the meeting was boring. Fill in the blank: The _ was boring.\nAnswer: the'` -> **meeting**
- order 1: `'The carnival was boring and the meeting was fun. Fill in the blank: The _ was fun.\nAnswer: the'` -> **meeting**
  - B: `'The carnival was boring and the meeting was fun. Fill in the blank: The _ was boring.\nAnswer: the'` -> **carnival**
- order 2: `'The meeting was boring and the carnival was fun. Fill in the blank: The _ was fun.\nAnswer: the'` -> **carnival**
  - B: `'The meeting was boring and the carnival was fun. Fill in the blank: The _ was boring.\nAnswer: the'` -> **meeting**
- order 2: `'The meeting was fun and the carnival was boring. Fill in the blank: The _ was fun.\nAnswer: the'` -> **meeting**
  - B: `'The meeting was fun and the carnival was boring. Fill in the blank: The _ was boring.\nAnswer: the'` -> **carnival**

## 26. ctrl|tea|snow|hot|cold (objects, ext)
- order 1: `'The tea was hot and the snow was cold. Fill in the blank: The _ was hot.\nAnswer: the'` -> **tea**
  - B: `'The tea was hot and the snow was cold. Fill in the blank: The _ was cold.\nAnswer: the'` -> **snow**
- order 1: `'The tea was cold and the snow was hot. Fill in the blank: The _ was hot.\nAnswer: the'` -> **snow**
  - B: `'The tea was cold and the snow was hot. Fill in the blank: The _ was cold.\nAnswer: the'` -> **tea**
- order 2: `'The snow was cold and the tea was hot. Fill in the blank: The _ was hot.\nAnswer: the'` -> **tea**
  - B: `'The snow was cold and the tea was hot. Fill in the blank: The _ was cold.\nAnswer: the'` -> **snow**
- order 2: `'The snow was hot and the tea was cold. Fill in the blank: The _ was hot.\nAnswer: the'` -> **snow**
  - B: `'The snow was hot and the tea was cold. Fill in the blank: The _ was cold.\nAnswer: the'` -> **tea**

## 27. ctrl|road|path|long|short (objects, ext)
- order 1: `'The road was long and the path was short. Fill in the blank: The _ was long.\nAnswer: the'` -> **road**
  - B: `'The road was long and the path was short. Fill in the blank: The _ was short.\nAnswer: the'` -> **path**
- order 1: `'The road was short and the path was long. Fill in the blank: The _ was long.\nAnswer: the'` -> **path**
  - B: `'The road was short and the path was long. Fill in the blank: The _ was short.\nAnswer: the'` -> **road**
- order 2: `'The path was short and the road was long. Fill in the blank: The _ was long.\nAnswer: the'` -> **road**
  - B: `'The path was short and the road was long. Fill in the blank: The _ was short.\nAnswer: the'` -> **path**
- order 2: `'The path was long and the road was short. Fill in the blank: The _ was long.\nAnswer: the'` -> **path**
  - B: `'The path was long and the road was short. Fill in the blank: The _ was short.\nAnswer: the'` -> **road**

## 28. ctrl|snow|mud|clean|dirty (objects, ext)
- order 1: `'The snow was clean and the mud was dirty. Fill in the blank: The _ was clean.\nAnswer: the'` -> **snow**
  - B: `'The snow was clean and the mud was dirty. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **mud**
- order 1: `'The snow was dirty and the mud was clean. Fill in the blank: The _ was clean.\nAnswer: the'` -> **mud**
  - B: `'The snow was dirty and the mud was clean. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **snow**
- order 2: `'The mud was dirty and the snow was clean. Fill in the blank: The _ was clean.\nAnswer: the'` -> **snow**
  - B: `'The mud was dirty and the snow was clean. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **mud**
- order 2: `'The mud was clean and the snow was dirty. Fill in the blank: The _ was clean.\nAnswer: the'` -> **mud**
  - B: `'The mud was clean and the snow was dirty. Fill in the blank: The _ was dirty.\nAnswer: the'` -> **snow**

## 29. 38DCH97KHJGG1T8T6YR77G20O6RQJ5 (objects, ext)
- order 1: `'The cake was delicious and the card was sentimental. Fill in the blank: The _ was delicious.\nAnswer: the'` -> **cake**
  - B: `'The cake was delicious and the card was sentimental. Fill in the blank: The _ was sentimental.\nAnswer: the'` -> **card**
- order 2: `'The card was sentimental and the cake was delicious. Fill in the blank: The _ was delicious.\nAnswer: the'` -> **cake**
  - B: `'The card was sentimental and the cake was delicious. Fill in the blank: The _ was sentimental.\nAnswer: the'` -> **card**

## 30. ctrl|planet|marble|big|small (objects, ext)
- order 1: `'The planet was big and the marble was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **planet**
  - B: `'The planet was big and the marble was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **marble**
- order 1: `'The planet was small and the marble was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **marble**
  - B: `'The planet was small and the marble was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **planet**
- order 2: `'The marble was small and the planet was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **planet**
  - B: `'The marble was small and the planet was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **marble**
- order 2: `'The marble was big and the planet was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **marble**
  - B: `'The marble was big and the planet was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **planet**

## 31. ctrl|rock|feather|heavy|light (objects, ext)
- order 1: `'The rock was heavy and the feather was light. Fill in the blank: The _ was heavy.\nAnswer: the'` -> **rock**
  - B: `'The rock was heavy and the feather was light. Fill in the blank: The _ was light.\nAnswer: the'` -> **feather**
- order 1: `'The rock was light and the feather was heavy. Fill in the blank: The _ was heavy.\nAnswer: the'` -> **feather**
  - B: `'The rock was light and the feather was heavy. Fill in the blank: The _ was light.\nAnswer: the'` -> **rock**
- order 2: `'The feather was light and the rock was heavy. Fill in the blank: The _ was heavy.\nAnswer: the'` -> **rock**
  - B: `'The feather was light and the rock was heavy. Fill in the blank: The _ was light.\nAnswer: the'` -> **feather**
- order 2: `'The feather was heavy and the rock was light. Fill in the blank: The _ was heavy.\nAnswer: the'` -> **feather**
  - B: `'The feather was heavy and the rock was light. Fill in the blank: The _ was light.\nAnswer: the'` -> **rock**

## 32. ctrl|cow|rat|big|small (objects, ext)
- order 1: `'The cow was big and the rat was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **cow**
  - B: `'The cow was big and the rat was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **rat**
- order 1: `'The cow was small and the rat was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **rat**
  - B: `'The cow was small and the rat was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **cow**
- order 2: `'The rat was small and the cow was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **cow**
  - B: `'The rat was small and the cow was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **rat**
- order 2: `'The rat was big and the cow was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **rat**
  - B: `'The rat was big and the cow was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **cow**

## 33. ctrl|jet|boat|fast|slow (objects, ext)
- order 1: `'The jet was fast and the boat was slow. Fill in the blank: The _ was fast.\nAnswer: the'` -> **jet**
  - B: `'The jet was fast and the boat was slow. Fill in the blank: The _ was slow.\nAnswer: the'` -> **boat**
- order 1: `'The jet was slow and the boat was fast. Fill in the blank: The _ was fast.\nAnswer: the'` -> **boat**
  - B: `'The jet was slow and the boat was fast. Fill in the blank: The _ was slow.\nAnswer: the'` -> **jet**
- order 2: `'The boat was slow and the jet was fast. Fill in the blank: The _ was fast.\nAnswer: the'` -> **jet**
  - B: `'The boat was slow and the jet was fast. Fill in the blank: The _ was slow.\nAnswer: the'` -> **boat**
- order 2: `'The boat was fast and the jet was slow. Fill in the blank: The _ was fast.\nAnswer: the'` -> **boat**
  - B: `'The boat was fast and the jet was slow. Fill in the blank: The _ was slow.\nAnswer: the'` -> **jet**

## 34. ctrl|wolf|sheep|dangerous|safe (objects, ext)
- order 1: `'The wolf was dangerous and the sheep was safe. Fill in the blank: The _ was dangerous.\nAnswer: the'` -> **wolf**
  - B: `'The wolf was dangerous and the sheep was safe. Fill in the blank: The _ was safe.\nAnswer: the'` -> **sheep**
- order 1: `'The wolf was safe and the sheep was dangerous. Fill in the blank: The _ was dangerous.\nAnswer: the'` -> **sheep**
  - B: `'The wolf was safe and the sheep was dangerous. Fill in the blank: The _ was safe.\nAnswer: the'` -> **wolf**
- order 2: `'The sheep was safe and the wolf was dangerous. Fill in the blank: The _ was dangerous.\nAnswer: the'` -> **wolf**
  - B: `'The sheep was safe and the wolf was dangerous. Fill in the blank: The _ was safe.\nAnswer: the'` -> **sheep**
- order 2: `'The sheep was dangerous and the wolf was safe. Fill in the blank: The _ was dangerous.\nAnswer: the'` -> **sheep**
  - B: `'The sheep was dangerous and the wolf was safe. Fill in the blank: The _ was safe.\nAnswer: the'` -> **wolf**

## 35. ctrl|tea|ice|warm|cold (objects, ext)
- order 1: `'The tea was warm and the ice was cold. Fill in the blank: The _ was warm.\nAnswer: the'` -> **tea**
  - B: `'The tea was warm and the ice was cold. Fill in the blank: The _ was cold.\nAnswer: the'` -> **ice**
- order 1: `'The tea was cold and the ice was warm. Fill in the blank: The _ was warm.\nAnswer: the'` -> **ice**
  - B: `'The tea was cold and the ice was warm. Fill in the blank: The _ was cold.\nAnswer: the'` -> **tea**
- order 2: `'The ice was cold and the tea was warm. Fill in the blank: The _ was warm.\nAnswer: the'` -> **tea**
  - B: `'The ice was cold and the tea was warm. Fill in the blank: The _ was cold.\nAnswer: the'` -> **ice**
- order 2: `'The ice was warm and the tea was cold. Fill in the blank: The _ was warm.\nAnswer: the'` -> **ice**
  - B: `'The ice was warm and the tea was cold. Fill in the blank: The _ was cold.\nAnswer: the'` -> **tea**

## 36. ctrl|steel|cotton|hard|soft (objects, ext)
- order 1: `'The steel was hard and the cotton was soft. Fill in the blank: The _ was hard.\nAnswer: the'` -> **steel**
  - B: `'The steel was hard and the cotton was soft. Fill in the blank: The _ was soft.\nAnswer: the'` -> **cotton**
- order 1: `'The steel was soft and the cotton was hard. Fill in the blank: The _ was hard.\nAnswer: the'` -> **cotton**
  - B: `'The steel was soft and the cotton was hard. Fill in the blank: The _ was soft.\nAnswer: the'` -> **steel**
- order 2: `'The cotton was soft and the steel was hard. Fill in the blank: The _ was hard.\nAnswer: the'` -> **steel**
  - B: `'The cotton was soft and the steel was hard. Fill in the blank: The _ was soft.\nAnswer: the'` -> **cotton**
- order 2: `'The cotton was hard and the steel was soft. Fill in the blank: The _ was hard.\nAnswer: the'` -> **cotton**
  - B: `'The cotton was hard and the steel was soft. Fill in the blank: The _ was soft.\nAnswer: the'` -> **steel**

## 37. ctrl|palace|cabin|big|small (objects, ext)
- order 1: `'The palace was big and the cabin was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **palace**
  - B: `'The palace was big and the cabin was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **cabin**
- order 1: `'The palace was small and the cabin was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **cabin**
  - B: `'The palace was small and the cabin was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **palace**
- order 2: `'The cabin was small and the palace was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **palace**
  - B: `'The cabin was small and the palace was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **cabin**
- order 2: `'The cabin was big and the palace was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **cabin**
  - B: `'The cabin was big and the palace was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **palace**

## 38. ctrl|car|bike|fast|slow (objects, ext)
- order 1: `'The car was fast and the bike was slow. Fill in the blank: The _ was fast.\nAnswer: the'` -> **car**
  - B: `'The car was fast and the bike was slow. Fill in the blank: The _ was slow.\nAnswer: the'` -> **bike**
- order 1: `'The car was slow and the bike was fast. Fill in the blank: The _ was fast.\nAnswer: the'` -> **bike**
  - B: `'The car was slow and the bike was fast. Fill in the blank: The _ was slow.\nAnswer: the'` -> **car**
- order 2: `'The bike was slow and the car was fast. Fill in the blank: The _ was fast.\nAnswer: the'` -> **car**
  - B: `'The bike was slow and the car was fast. Fill in the blank: The _ was slow.\nAnswer: the'` -> **bike**
- order 2: `'The bike was fast and the car was slow. Fill in the blank: The _ was fast.\nAnswer: the'` -> **bike**
  - B: `'The bike was fast and the car was slow. Fill in the blank: The _ was slow.\nAnswer: the'` -> **car**

## 39. ctrl|ocean|stream|wide|narrow (objects, ext)
- order 1: `'The ocean was wide and the stream was narrow. Fill in the blank: The _ was wide.\nAnswer: the'` -> **ocean**
  - B: `'The ocean was wide and the stream was narrow. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **stream**
- order 1: `'The ocean was narrow and the stream was wide. Fill in the blank: The _ was wide.\nAnswer: the'` -> **stream**
  - B: `'The ocean was narrow and the stream was wide. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **ocean**
- order 2: `'The stream was narrow and the ocean was wide. Fill in the blank: The _ was wide.\nAnswer: the'` -> **ocean**
  - B: `'The stream was narrow and the ocean was wide. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **stream**
- order 2: `'The stream was wide and the ocean was narrow. Fill in the blank: The _ was wide.\nAnswer: the'` -> **stream**
  - B: `'The stream was wide and the ocean was narrow. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **ocean**

## 40. ctrl|castle|hut|big|small (objects, ext)
- order 1: `'The castle was big and the hut was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **castle**
  - B: `'The castle was big and the hut was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **hut**
- order 1: `'The castle was small and the hut was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **hut**
  - B: `'The castle was small and the hut was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **castle**
- order 2: `'The hut was small and the castle was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **castle**
  - B: `'The hut was small and the castle was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **hut**
- order 2: `'The hut was big and the castle was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **hut**
  - B: `'The hut was big and the castle was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **castle**

## 41. ctrl|iron|velvet|hard|soft (objects, ext)
- order 1: `'The iron was hard and the velvet was soft. Fill in the blank: The _ was hard.\nAnswer: the'` -> **iron**
  - B: `'The iron was hard and the velvet was soft. Fill in the blank: The _ was soft.\nAnswer: the'` -> **velvet**
- order 1: `'The iron was soft and the velvet was hard. Fill in the blank: The _ was hard.\nAnswer: the'` -> **velvet**
  - B: `'The iron was soft and the velvet was hard. Fill in the blank: The _ was soft.\nAnswer: the'` -> **iron**
- order 2: `'The velvet was soft and the iron was hard. Fill in the blank: The _ was hard.\nAnswer: the'` -> **iron**
  - B: `'The velvet was soft and the iron was hard. Fill in the blank: The _ was soft.\nAnswer: the'` -> **velvet**
- order 2: `'The velvet was hard and the iron was soft. Fill in the blank: The _ was hard.\nAnswer: the'` -> **velvet**
  - B: `'The velvet was hard and the iron was soft. Fill in the blank: The _ was soft.\nAnswer: the'` -> **iron**

## 42. ctrl|lake|dust|wet|dry (objects, ext)
- order 1: `'The lake was wet and the dust was dry. Fill in the blank: The _ was wet.\nAnswer: the'` -> **lake**
  - B: `'The lake was wet and the dust was dry. Fill in the blank: The _ was dry.\nAnswer: the'` -> **dust**
- order 1: `'The lake was dry and the dust was wet. Fill in the blank: The _ was wet.\nAnswer: the'` -> **dust**
  - B: `'The lake was dry and the dust was wet. Fill in the blank: The _ was dry.\nAnswer: the'` -> **lake**
- order 2: `'The dust was dry and the lake was wet. Fill in the blank: The _ was wet.\nAnswer: the'` -> **lake**
  - B: `'The dust was dry and the lake was wet. Fill in the blank: The _ was dry.\nAnswer: the'` -> **dust**
- order 2: `'The dust was wet and the lake was dry. Fill in the blank: The _ was wet.\nAnswer: the'` -> **dust**
  - B: `'The dust was wet and the lake was dry. Fill in the blank: The _ was dry.\nAnswer: the'` -> **lake**

## 43. ctrl|horse|ant|big|small (objects, ext)
- order 1: `'The horse was big and the ant was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **horse**
  - B: `'The horse was big and the ant was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **ant**
- order 1: `'The horse was small and the ant was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **ant**
  - B: `'The horse was small and the ant was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **horse**
- order 2: `'The ant was small and the horse was big. Fill in the blank: The _ was big.\nAnswer: the'` -> **horse**
  - B: `'The ant was small and the horse was big. Fill in the blank: The _ was small.\nAnswer: the'` -> **ant**
- order 2: `'The ant was big and the horse was small. Fill in the blank: The _ was big.\nAnswer: the'` -> **ant**
  - B: `'The ant was big and the horse was small. Fill in the blank: The _ was small.\nAnswer: the'` -> **horse**

## 44. ctrl|ocean|desert|wet|dry (objects, ext)
- order 1: `'The ocean was wet and the desert was dry. Fill in the blank: The _ was wet.\nAnswer: the'` -> **ocean**
  - B: `'The ocean was wet and the desert was dry. Fill in the blank: The _ was dry.\nAnswer: the'` -> **desert**
- order 1: `'The ocean was dry and the desert was wet. Fill in the blank: The _ was wet.\nAnswer: the'` -> **desert**
  - B: `'The ocean was dry and the desert was wet. Fill in the blank: The _ was dry.\nAnswer: the'` -> **ocean**
- order 2: `'The desert was dry and the ocean was wet. Fill in the blank: The _ was wet.\nAnswer: the'` -> **ocean**
  - B: `'The desert was dry and the ocean was wet. Fill in the blank: The _ was dry.\nAnswer: the'` -> **desert**
- order 2: `'The desert was wet and the ocean was dry. Fill in the blank: The _ was wet.\nAnswer: the'` -> **desert**
  - B: `'The desert was wet and the ocean was dry. Fill in the blank: The _ was dry.\nAnswer: the'` -> **ocean**

## 45. ctrl|stadium|graveyard|full|empty (objects, ext)
- order 1: `'The stadium was full and the graveyard was empty. Fill in the blank: The _ was full.\nAnswer: the'` -> **stadium**
  - B: `'The stadium was full and the graveyard was empty. Fill in the blank: The _ was empty.\nAnswer: the'` -> **graveyard**
- order 1: `'The stadium was empty and the graveyard was full. Fill in the blank: The _ was full.\nAnswer: the'` -> **graveyard**
  - B: `'The stadium was empty and the graveyard was full. Fill in the blank: The _ was empty.\nAnswer: the'` -> **stadium**
- order 2: `'The graveyard was empty and the stadium was full. Fill in the blank: The _ was full.\nAnswer: the'` -> **stadium**
  - B: `'The graveyard was empty and the stadium was full. Fill in the blank: The _ was empty.\nAnswer: the'` -> **graveyard**
- order 2: `'The graveyard was full and the stadium was empty. Fill in the blank: The _ was full.\nAnswer: the'` -> **graveyard**
  - B: `'The graveyard was full and the stadium was empty. Fill in the blank: The _ was empty.\nAnswer: the'` -> **stadium**

## 46. ctrl|road|path|wide|narrow (objects, ext)
- order 1: `'The road was wide and the path was narrow. Fill in the blank: The _ was wide.\nAnswer: the'` -> **road**
  - B: `'The road was wide and the path was narrow. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **path**
- order 1: `'The road was narrow and the path was wide. Fill in the blank: The _ was wide.\nAnswer: the'` -> **path**
  - B: `'The road was narrow and the path was wide. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **road**
- order 2: `'The path was narrow and the road was wide. Fill in the blank: The _ was wide.\nAnswer: the'` -> **road**
  - B: `'The path was narrow and the road was wide. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **path**
- order 2: `'The path was wide and the road was narrow. Fill in the blank: The _ was wide.\nAnswer: the'` -> **path**
  - B: `'The path was wide and the road was narrow. Fill in the blank: The _ was narrow.\nAnswer: the'` -> **road**

## 47. ctrl|train|wagon|fast|slow (objects, ext)
- order 1: `'The train was fast and the wagon was slow. Fill in the blank: The _ was fast.\nAnswer: the'` -> **train**
  - B: `'The train was fast and the wagon was slow. Fill in the blank: The _ was slow.\nAnswer: the'` -> **wagon**
- order 1: `'The train was slow and the wagon was fast. Fill in the blank: The _ was fast.\nAnswer: the'` -> **wagon**
  - B: `'The train was slow and the wagon was fast. Fill in the blank: The _ was slow.\nAnswer: the'` -> **train**
- order 2: `'The wagon was slow and the train was fast. Fill in the blank: The _ was fast.\nAnswer: the'` -> **train**
  - B: `'The wagon was slow and the train was fast. Fill in the blank: The _ was slow.\nAnswer: the'` -> **wagon**
- order 2: `'The wagon was fast and the train was slow. Fill in the blank: The _ was fast.\nAnswer: the'` -> **wagon**
  - B: `'The wagon was fast and the train was slow. Fill in the blank: The _ was slow.\nAnswer: the'` -> **train**

## 48. 3HYV4299H2A2K9M563ZGO948PS98ER (objects, core)
- order 1: `'The head was bald and the crowd was vast. Fill in the blank: The _ was bald.\nAnswer: the'` -> **head**
  - B: `'The head was bald and the crowd was vast. Fill in the blank: The _ was vast.\nAnswer: the'` -> **crowd**
- order 2: `'The crowd was vast and the head was bald. Fill in the blank: The _ was bald.\nAnswer: the'` -> **head**
  - B: `'The crowd was vast and the head was bald. Fill in the blank: The _ was vast.\nAnswer: the'` -> **crowd**

## 49. ctrl|airplane|bee|large|tiny (objects, ext)
- order 1: `'The airplane was large and the bee was tiny. Fill in the blank: The _ was large.\nAnswer: the'` -> **airplane**
  - B: `'The airplane was large and the bee was tiny. Fill in the blank: The _ was tiny.\nAnswer: the'` -> **bee**
- order 1: `'The airplane was tiny and the bee was large. Fill in the blank: The _ was large.\nAnswer: the'` -> **bee**
  - B: `'The airplane was tiny and the bee was large. Fill in the blank: The _ was tiny.\nAnswer: the'` -> **airplane**
- order 2: `'The bee was tiny and the airplane was large. Fill in the blank: The _ was large.\nAnswer: the'` -> **airplane**
  - B: `'The bee was tiny and the airplane was large. Fill in the blank: The _ was tiny.\nAnswer: the'` -> **bee**
- order 2: `'The bee was large and the airplane was tiny. Fill in the blank: The _ was large.\nAnswer: the'` -> **bee**
  - B: `'The bee was large and the airplane was tiny. Fill in the blank: The _ was tiny.\nAnswer: the'` -> **airplane**

## 50. ctrl|highway|graveyard|noisy|silent (objects, ext)
- order 1: `'The highway was noisy and the graveyard was silent. Fill in the blank: The _ was noisy.\nAnswer: the'` -> **highway**
  - B: `'The highway was noisy and the graveyard was silent. Fill in the blank: The _ was silent.\nAnswer: the'` -> **graveyard**
- order 1: `'The highway was silent and the graveyard was noisy. Fill in the blank: The _ was noisy.\nAnswer: the'` -> **graveyard**
  - B: `'The highway was silent and the graveyard was noisy. Fill in the blank: The _ was silent.\nAnswer: the'` -> **highway**
- order 2: `'The graveyard was silent and the highway was noisy. Fill in the blank: The _ was noisy.\nAnswer: the'` -> **highway**
  - B: `'The graveyard was silent and the highway was noisy. Fill in the blank: The _ was silent.\nAnswer: the'` -> **graveyard**
- order 2: `'The graveyard was noisy and the highway was silent. Fill in the blank: The _ was noisy.\nAnswer: the'` -> **graveyard**
  - B: `'The graveyard was noisy and the highway was silent. Fill in the blank: The _ was silent.\nAnswer: the'` -> **highway**
