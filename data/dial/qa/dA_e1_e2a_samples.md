# dA_e1_e2a: 50 seeded samples (qwen-family texts; random_state 0)

Each item: prompt A -> answer A | prompt B -> answer B (STR at token positions str_pos). Orders shown where built.

## 1. ctrl|party|funeral|happy|sad (objects, ext)
- order 1: `'The party was happy and the funeral was sad, so the party was'` -> **happy**
  - B: `'The party was happy and the funeral was sad, so the funeral was'` -> **sad**
- order 1: `'The party was sad and the funeral was happy, so the party was'` -> **sad**
  - B: `'The party was sad and the funeral was happy, so the funeral was'` -> **happy**
- order 2: `'The funeral was sad and the party was happy, so the party was'` -> **happy**
  - B: `'The funeral was sad and the party was happy, so the funeral was'` -> **sad**
- order 2: `'The funeral was happy and the party was sad, so the party was'` -> **sad**
  - B: `'The funeral was happy and the party was sad, so the funeral was'` -> **happy**

## 2. ctrl|birthday|funeral|happy|sad (objects, ext)
- order 1: `'The birthday was happy and the funeral was sad, so the birthday was'` -> **happy**
  - B: `'The birthday was happy and the funeral was sad, so the funeral was'` -> **sad**
- order 1: `'The birthday was sad and the funeral was happy, so the birthday was'` -> **sad**
  - B: `'The birthday was sad and the funeral was happy, so the funeral was'` -> **happy**
- order 2: `'The funeral was sad and the birthday was happy, so the birthday was'` -> **happy**
  - B: `'The funeral was sad and the birthday was happy, so the funeral was'` -> **sad**
- order 2: `'The funeral was happy and the birthday was sad, so the birthday was'` -> **sad**
  - B: `'The funeral was happy and the birthday was sad, so the funeral was'` -> **happy**

## 3. ctrl|cloud|mud|high|low (objects, ext)
- order 1: `'The cloud was high and the mud was low, so the cloud was'` -> **high**
  - B: `'The cloud was high and the mud was low, so the mud was'` -> **low**
- order 1: `'The cloud was low and the mud was high, so the cloud was'` -> **low**
  - B: `'The cloud was low and the mud was high, so the mud was'` -> **high**
- order 2: `'The mud was low and the cloud was high, so the cloud was'` -> **high**
  - B: `'The mud was low and the cloud was high, so the mud was'` -> **low**
- order 2: `'The mud was high and the cloud was low, so the cloud was'` -> **low**
  - B: `'The mud was high and the cloud was low, so the mud was'` -> **high**

## 4. ctrl|stone|wool|hard|soft (objects, ext)
- order 1: `'The stone was hard and the wool was soft, so the stone was'` -> **hard**
  - B: `'The stone was hard and the wool was soft, so the wool was'` -> **soft**
- order 1: `'The stone was soft and the wool was hard, so the stone was'` -> **soft**
  - B: `'The stone was soft and the wool was hard, so the wool was'` -> **hard**
- order 2: `'The wool was soft and the stone was hard, so the stone was'` -> **hard**
  - B: `'The wool was soft and the stone was hard, so the wool was'` -> **soft**
- order 2: `'The wool was hard and the stone was soft, so the stone was'` -> **soft**
  - B: `'The wool was hard and the stone was soft, so the wool was'` -> **hard**

## 5. ctrl|song|funeral|happy|sad (objects, ext)
- order 1: `'The song was happy and the funeral was sad, so the song was'` -> **happy**
  - B: `'The song was happy and the funeral was sad, so the funeral was'` -> **sad**
- order 1: `'The song was sad and the funeral was happy, so the song was'` -> **sad**
  - B: `'The song was sad and the funeral was happy, so the funeral was'` -> **happy**
- order 2: `'The funeral was sad and the song was happy, so the song was'` -> **happy**
  - B: `'The funeral was sad and the song was happy, so the funeral was'` -> **sad**
- order 2: `'The funeral was happy and the song was sad, so the song was'` -> **sad**
  - B: `'The funeral was happy and the song was sad, so the funeral was'` -> **happy**

## 6. ctrl|vacation|commute|fun|boring (objects, ext)
- order 1: `'The vacation was fun and the commute was boring, so the vacation was'` -> **fun**
  - B: `'The vacation was fun and the commute was boring, so the commute was'` -> **boring**
- order 1: `'The vacation was boring and the commute was fun, so the vacation was'` -> **boring**
  - B: `'The vacation was boring and the commute was fun, so the commute was'` -> **fun**
- order 2: `'The commute was boring and the vacation was fun, so the vacation was'` -> **fun**
  - B: `'The commute was boring and the vacation was fun, so the commute was'` -> **boring**
- order 2: `'The commute was fun and the vacation was boring, so the vacation was'` -> **boring**
  - B: `'The commute was fun and the vacation was boring, so the commute was'` -> **fun**

## 7. ctrl|stove|freezer|hot|cold (objects, ext)
- order 1: `'The stove was hot and the freezer was cold, so the stove was'` -> **hot**
  - B: `'The stove was hot and the freezer was cold, so the freezer was'` -> **cold**
- order 1: `'The stove was cold and the freezer was hot, so the stove was'` -> **cold**
  - B: `'The stove was cold and the freezer was hot, so the freezer was'` -> **hot**
- order 2: `'The freezer was cold and the stove was hot, so the stove was'` -> **hot**
  - B: `'The freezer was cold and the stove was hot, so the freezer was'` -> **cold**
- order 2: `'The freezer was hot and the stove was cold, so the stove was'` -> **cold**
  - B: `'The freezer was hot and the stove was cold, so the freezer was'` -> **hot**

## 8. ctrl|salsa|yogurt|spicy|mild (objects, ext)
- order 1: `'The salsa was spicy and the yogurt was mild, so the salsa was'` -> **spicy**
  - B: `'The salsa was spicy and the yogurt was mild, so the yogurt was'` -> **mild**
- order 1: `'The salsa was mild and the yogurt was spicy, so the salsa was'` -> **mild**
  - B: `'The salsa was mild and the yogurt was spicy, so the yogurt was'` -> **spicy**
- order 2: `'The yogurt was mild and the salsa was spicy, so the salsa was'` -> **spicy**
  - B: `'The yogurt was mild and the salsa was spicy, so the yogurt was'` -> **mild**
- order 2: `'The yogurt was spicy and the salsa was mild, so the salsa was'` -> **mild**
  - B: `'The yogurt was spicy and the salsa was mild, so the yogurt was'` -> **spicy**

## 9. ctrl|concert|library|loud|quiet (objects, ext)
- order 1: `'The concert was loud and the library was quiet, so the concert was'` -> **loud**
  - B: `'The concert was loud and the library was quiet, so the library was'` -> **quiet**
- order 1: `'The concert was quiet and the library was loud, so the concert was'` -> **quiet**
  - B: `'The concert was quiet and the library was loud, so the library was'` -> **loud**
- order 2: `'The library was quiet and the concert was loud, so the concert was'` -> **loud**
  - B: `'The library was quiet and the concert was loud, so the library was'` -> **quiet**
- order 2: `'The library was loud and the concert was quiet, so the concert was'` -> **quiet**
  - B: `'The library was loud and the concert was quiet, so the library was'` -> **loud**

## 10. ctrl|giant|child|tall|short (objects, ext)
- order 1: `'The giant was tall and the child was short, so the giant was'` -> **tall**
  - B: `'The giant was tall and the child was short, so the child was'` -> **short**
- order 1: `'The giant was short and the child was tall, so the giant was'` -> **short**
  - B: `'The giant was short and the child was tall, so the child was'` -> **tall**
- order 2: `'The child was short and the giant was tall, so the giant was'` -> **tall**
  - B: `'The child was short and the giant was tall, so the child was'` -> **short**
- order 2: `'The child was tall and the giant was short, so the giant was'` -> **short**
  - B: `'The child was tall and the giant was short, so the child was'` -> **tall**

## 11. ctrl|hospital|dump|clean|dirty (objects, ext)
- order 1: `'The hospital was clean and the dump was dirty, so the hospital was'` -> **clean**
  - B: `'The hospital was clean and the dump was dirty, so the dump was'` -> **dirty**
- order 1: `'The hospital was dirty and the dump was clean, so the hospital was'` -> **dirty**
  - B: `'The hospital was dirty and the dump was clean, so the dump was'` -> **clean**
- order 2: `'The dump was dirty and the hospital was clean, so the hospital was'` -> **clean**
  - B: `'The dump was dirty and the hospital was clean, so the dump was'` -> **dirty**
- order 2: `'The dump was clean and the hospital was dirty, so the hospital was'` -> **dirty**
  - B: `'The dump was clean and the hospital was dirty, so the dump was'` -> **clean**

## 12. ctrl|candy|lemon|sweet|sour (objects, ext)
- order 1: `'The candy was sweet and the lemon was sour, so the candy was'` -> **sweet**
  - B: `'The candy was sweet and the lemon was sour, so the lemon was'` -> **sour**
- order 1: `'The candy was sour and the lemon was sweet, so the candy was'` -> **sour**
  - B: `'The candy was sour and the lemon was sweet, so the lemon was'` -> **sweet**
- order 2: `'The lemon was sour and the candy was sweet, so the candy was'` -> **sweet**
  - B: `'The lemon was sour and the candy was sweet, so the lemon was'` -> **sour**
- order 2: `'The lemon was sweet and the candy was sour, so the candy was'` -> **sour**
  - B: `'The lemon was sweet and the candy was sour, so the lemon was'` -> **sweet**

## 13. ctrl|lake|desert|wet|dry (objects, ext)
- order 1: `'The lake was wet and the desert was dry, so the lake was'` -> **wet**
  - B: `'The lake was wet and the desert was dry, so the desert was'` -> **dry**
- order 1: `'The lake was dry and the desert was wet, so the lake was'` -> **dry**
  - B: `'The lake was dry and the desert was wet, so the desert was'` -> **wet**
- order 2: `'The desert was dry and the lake was wet, so the lake was'` -> **wet**
  - B: `'The desert was dry and the lake was wet, so the desert was'` -> **dry**
- order 2: `'The desert was wet and the lake was dry, so the lake was'` -> **dry**
  - B: `'The desert was wet and the lake was dry, so the desert was'` -> **wet**

## 14. ctrl|knife|spoon|sharp|dull (objects, ext)
- order 1: `'The knife was sharp and the spoon was dull, so the knife was'` -> **sharp**
  - B: `'The knife was sharp and the spoon was dull, so the spoon was'` -> **dull**
- order 1: `'The knife was dull and the spoon was sharp, so the knife was'` -> **dull**
  - B: `'The knife was dull and the spoon was sharp, so the spoon was'` -> **sharp**
- order 2: `'The spoon was dull and the knife was sharp, so the knife was'` -> **sharp**
  - B: `'The spoon was dull and the knife was sharp, so the spoon was'` -> **dull**
- order 2: `'The spoon was sharp and the knife was dull, so the knife was'` -> **dull**
  - B: `'The spoon was sharp and the knife was dull, so the spoon was'` -> **sharp**

## 15. ctrl|soup|ice|hot|cold (objects, ext)
- order 1: `'The soup was hot and the ice was cold, so the soup was'` -> **hot**
  - B: `'The soup was hot and the ice was cold, so the ice was'` -> **cold**
- order 1: `'The soup was cold and the ice was hot, so the soup was'` -> **cold**
  - B: `'The soup was cold and the ice was hot, so the ice was'` -> **hot**
- order 2: `'The ice was cold and the soup was hot, so the soup was'` -> **hot**
  - B: `'The ice was cold and the soup was hot, so the ice was'` -> **cold**
- order 2: `'The ice was hot and the soup was cold, so the soup was'` -> **cold**
  - B: `'The ice was hot and the soup was cold, so the ice was'` -> **hot**

## 16. ctrl|bear|rabbit|big|small (objects, ext)
- order 1: `'The bear was big and the rabbit was small, so the bear was'` -> **big**
  - B: `'The bear was big and the rabbit was small, so the rabbit was'` -> **small**
- order 1: `'The bear was small and the rabbit was big, so the bear was'` -> **small**
  - B: `'The bear was small and the rabbit was big, so the rabbit was'` -> **big**
- order 2: `'The rabbit was small and the bear was big, so the bear was'` -> **big**
  - B: `'The rabbit was small and the bear was big, so the rabbit was'` -> **small**
- order 2: `'The rabbit was big and the bear was small, so the bear was'` -> **small**
  - B: `'The rabbit was big and the bear was small, so the rabbit was'` -> **big**

## 17. ctrl|egg|rock|fragile|sturdy (objects, ext)
- order 1: `'The egg was fragile and the rock was sturdy, so the egg was'` -> **fragile**
  - B: `'The egg was fragile and the rock was sturdy, so the rock was'` -> **sturdy**
- order 1: `'The egg was sturdy and the rock was fragile, so the egg was'` -> **sturdy**
  - B: `'The egg was sturdy and the rock was fragile, so the rock was'` -> **fragile**
- order 2: `'The rock was sturdy and the egg was fragile, so the egg was'` -> **fragile**
  - B: `'The rock was sturdy and the egg was fragile, so the rock was'` -> **sturdy**
- order 2: `'The rock was fragile and the egg was sturdy, so the egg was'` -> **sturdy**
  - B: `'The rock was fragile and the egg was sturdy, so the rock was'` -> **fragile**

## 18. ctrl|party|lecture|fun|boring (objects, ext)
- order 1: `'The party was fun and the lecture was boring, so the party was'` -> **fun**
  - B: `'The party was fun and the lecture was boring, so the lecture was'` -> **boring**
- order 1: `'The party was boring and the lecture was fun, so the party was'` -> **boring**
  - B: `'The party was boring and the lecture was fun, so the lecture was'` -> **fun**
- order 2: `'The lecture was boring and the party was fun, so the party was'` -> **fun**
  - B: `'The lecture was boring and the party was fun, so the lecture was'` -> **boring**
- order 2: `'The lecture was fun and the party was boring, so the party was'` -> **boring**
  - B: `'The lecture was fun and the party was boring, so the lecture was'` -> **fun**

## 19. ctrl|soap|dirt|clean|dirty (objects, ext)
- order 1: `'The soap was clean and the dirt was dirty, so the soap was'` -> **clean**
  - B: `'The soap was clean and the dirt was dirty, so the dirt was'` -> **dirty**
- order 1: `'The soap was dirty and the dirt was clean, so the soap was'` -> **dirty**
  - B: `'The soap was dirty and the dirt was clean, so the dirt was'` -> **clean**
- order 2: `'The dirt was dirty and the soap was clean, so the soap was'` -> **clean**
  - B: `'The dirt was dirty and the soap was clean, so the dirt was'` -> **dirty**
- order 2: `'The dirt was clean and the soap was dirty, so the soap was'` -> **dirty**
  - B: `'The dirt was clean and the soap was dirty, so the dirt was'` -> **clean**

## 20. ctrl|sword|stick|sharp|blunt (objects, ext)
- order 1: `'The sword was sharp and the stick was blunt, so the sword was'` -> **sharp**
  - B: `'The sword was sharp and the stick was blunt, so the stick was'` -> **blunt**
- order 1: `'The sword was blunt and the stick was sharp, so the sword was'` -> **blunt**
  - B: `'The sword was blunt and the stick was sharp, so the stick was'` -> **sharp**
- order 2: `'The stick was blunt and the sword was sharp, so the sword was'` -> **sharp**
  - B: `'The stick was blunt and the sword was sharp, so the stick was'` -> **blunt**
- order 2: `'The stick was sharp and the sword was blunt, so the sword was'` -> **blunt**
  - B: `'The stick was sharp and the sword was blunt, so the stick was'` -> **sharp**

## 21. ctrl|kale|cookie|bitter|sweet (objects, ext)
- order 1: `'The kale was bitter and the cookie was sweet, so the kale was'` -> **bitter**
  - B: `'The kale was bitter and the cookie was sweet, so the cookie was'` -> **sweet**
- order 1: `'The kale was sweet and the cookie was bitter, so the kale was'` -> **sweet**
  - B: `'The kale was sweet and the cookie was bitter, so the cookie was'` -> **bitter**
- order 2: `'The cookie was sweet and the kale was bitter, so the kale was'` -> **bitter**
  - B: `'The cookie was sweet and the kale was bitter, so the cookie was'` -> **sweet**
- order 2: `'The cookie was bitter and the kale was sweet, so the kale was'` -> **sweet**
  - B: `'The cookie was bitter and the kale was sweet, so the cookie was'` -> **bitter**

## 22. ctrl|movie|exam|fun|boring (objects, ext)
- order 1: `'The movie was fun and the exam was boring, so the movie was'` -> **fun**
  - B: `'The movie was fun and the exam was boring, so the exam was'` -> **boring**
- order 1: `'The movie was boring and the exam was fun, so the movie was'` -> **boring**
  - B: `'The movie was boring and the exam was fun, so the exam was'` -> **fun**
- order 2: `'The exam was boring and the movie was fun, so the movie was'` -> **fun**
  - B: `'The exam was boring and the movie was fun, so the exam was'` -> **boring**
- order 2: `'The exam was fun and the movie was boring, so the movie was'` -> **boring**
  - B: `'The exam was fun and the movie was boring, so the exam was'` -> **fun**

## 23. ctrl|river|path|wide|narrow (objects, ext)
- order 1: `'The river was wide and the path was narrow, so the river was'` -> **wide**
  - B: `'The river was wide and the path was narrow, so the path was'` -> **narrow**
- order 1: `'The river was narrow and the path was wide, so the river was'` -> **narrow**
  - B: `'The river was narrow and the path was wide, so the path was'` -> **wide**
- order 2: `'The path was narrow and the river was wide, so the river was'` -> **wide**
  - B: `'The path was narrow and the river was wide, so the path was'` -> **narrow**
- order 2: `'The path was wide and the river was narrow, so the river was'` -> **narrow**
  - B: `'The path was wide and the river was narrow, so the path was'` -> **wide**

## 24. 3O2Y2UIUCQSZRZPVQ54P4YAQ7QOFKP (objects, ext)
- order 1: `'The bedroom was too large and the house was too small, so the bedroom was too'` -> **large**
  - B: `'The bedroom was too large and the house was too small, so the house was too'` -> **small**
- order 2: `'The house was too small and the bedroom was too large, so the bedroom was too'` -> **large**
  - B: `'The house was too small and the bedroom was too large, so the house was too'` -> **small**

## 25. ctrl|carnival|meeting|fun|boring (objects, ext)
- order 1: `'The carnival was fun and the meeting was boring, so the carnival was'` -> **fun**
  - B: `'The carnival was fun and the meeting was boring, so the meeting was'` -> **boring**
- order 1: `'The carnival was boring and the meeting was fun, so the carnival was'` -> **boring**
  - B: `'The carnival was boring and the meeting was fun, so the meeting was'` -> **fun**
- order 2: `'The meeting was boring and the carnival was fun, so the carnival was'` -> **fun**
  - B: `'The meeting was boring and the carnival was fun, so the meeting was'` -> **boring**
- order 2: `'The meeting was fun and the carnival was boring, so the carnival was'` -> **boring**
  - B: `'The meeting was fun and the carnival was boring, so the meeting was'` -> **fun**

## 26. ctrl|tea|snow|hot|cold (objects, ext)
- order 1: `'The tea was hot and the snow was cold, so the tea was'` -> **hot**
  - B: `'The tea was hot and the snow was cold, so the snow was'` -> **cold**
- order 1: `'The tea was cold and the snow was hot, so the tea was'` -> **cold**
  - B: `'The tea was cold and the snow was hot, so the snow was'` -> **hot**
- order 2: `'The snow was cold and the tea was hot, so the tea was'` -> **hot**
  - B: `'The snow was cold and the tea was hot, so the snow was'` -> **cold**
- order 2: `'The snow was hot and the tea was cold, so the tea was'` -> **cold**
  - B: `'The snow was hot and the tea was cold, so the snow was'` -> **hot**

## 27. ctrl|road|path|long|short (objects, ext)
- order 1: `'The road was long and the path was short, so the road was'` -> **long**
  - B: `'The road was long and the path was short, so the path was'` -> **short**
- order 1: `'The road was short and the path was long, so the road was'` -> **short**
  - B: `'The road was short and the path was long, so the path was'` -> **long**
- order 2: `'The path was short and the road was long, so the road was'` -> **long**
  - B: `'The path was short and the road was long, so the path was'` -> **short**
- order 2: `'The path was long and the road was short, so the road was'` -> **short**
  - B: `'The path was long and the road was short, so the path was'` -> **long**

## 28. ctrl|snow|mud|clean|dirty (objects, ext)
- order 1: `'The snow was clean and the mud was dirty, so the snow was'` -> **clean**
  - B: `'The snow was clean and the mud was dirty, so the mud was'` -> **dirty**
- order 1: `'The snow was dirty and the mud was clean, so the snow was'` -> **dirty**
  - B: `'The snow was dirty and the mud was clean, so the mud was'` -> **clean**
- order 2: `'The mud was dirty and the snow was clean, so the snow was'` -> **clean**
  - B: `'The mud was dirty and the snow was clean, so the mud was'` -> **dirty**
- order 2: `'The mud was clean and the snow was dirty, so the snow was'` -> **dirty**
  - B: `'The mud was clean and the snow was dirty, so the mud was'` -> **clean**

## 29. 38DCH97KHJGG1T8T6YR77G20O6RQJ5 (objects, ext)
- order 1: `'The cake was delicious and the card was sentimental, so the cake was'` -> **delicious**
  - B: `'The cake was delicious and the card was sentimental, so the card was'` -> **sentimental**
- order 2: `'The card was sentimental and the cake was delicious, so the cake was'` -> **delicious**
  - B: `'The card was sentimental and the cake was delicious, so the card was'` -> **sentimental**

## 30. ctrl|planet|marble|big|small (objects, ext)
- order 1: `'The planet was big and the marble was small, so the planet was'` -> **big**
  - B: `'The planet was big and the marble was small, so the marble was'` -> **small**
- order 1: `'The planet was small and the marble was big, so the planet was'` -> **small**
  - B: `'The planet was small and the marble was big, so the marble was'` -> **big**
- order 2: `'The marble was small and the planet was big, so the planet was'` -> **big**
  - B: `'The marble was small and the planet was big, so the marble was'` -> **small**
- order 2: `'The marble was big and the planet was small, so the planet was'` -> **small**
  - B: `'The marble was big and the planet was small, so the marble was'` -> **big**

## 31. ctrl|rock|feather|heavy|light (objects, ext)
- order 1: `'The rock was heavy and the feather was light, so the rock was'` -> **heavy**
  - B: `'The rock was heavy and the feather was light, so the feather was'` -> **light**
- order 1: `'The rock was light and the feather was heavy, so the rock was'` -> **light**
  - B: `'The rock was light and the feather was heavy, so the feather was'` -> **heavy**
- order 2: `'The feather was light and the rock was heavy, so the rock was'` -> **heavy**
  - B: `'The feather was light and the rock was heavy, so the feather was'` -> **light**
- order 2: `'The feather was heavy and the rock was light, so the rock was'` -> **light**
  - B: `'The feather was heavy and the rock was light, so the feather was'` -> **heavy**

## 32. ctrl|cow|rat|big|small (objects, ext)
- order 1: `'The cow was big and the rat was small, so the cow was'` -> **big**
  - B: `'The cow was big and the rat was small, so the rat was'` -> **small**
- order 1: `'The cow was small and the rat was big, so the cow was'` -> **small**
  - B: `'The cow was small and the rat was big, so the rat was'` -> **big**
- order 2: `'The rat was small and the cow was big, so the cow was'` -> **big**
  - B: `'The rat was small and the cow was big, so the rat was'` -> **small**
- order 2: `'The rat was big and the cow was small, so the cow was'` -> **small**
  - B: `'The rat was big and the cow was small, so the rat was'` -> **big**

## 33. ctrl|jet|boat|fast|slow (objects, ext)
- order 1: `'The jet was fast and the boat was slow, so the jet was'` -> **fast**
  - B: `'The jet was fast and the boat was slow, so the boat was'` -> **slow**
- order 1: `'The jet was slow and the boat was fast, so the jet was'` -> **slow**
  - B: `'The jet was slow and the boat was fast, so the boat was'` -> **fast**
- order 2: `'The boat was slow and the jet was fast, so the jet was'` -> **fast**
  - B: `'The boat was slow and the jet was fast, so the boat was'` -> **slow**
- order 2: `'The boat was fast and the jet was slow, so the jet was'` -> **slow**
  - B: `'The boat was fast and the jet was slow, so the boat was'` -> **fast**

## 34. ctrl|wolf|sheep|dangerous|safe (objects, ext)
- order 1: `'The wolf was dangerous and the sheep was safe, so the wolf was'` -> **dangerous**
  - B: `'The wolf was dangerous and the sheep was safe, so the sheep was'` -> **safe**
- order 1: `'The wolf was safe and the sheep was dangerous, so the wolf was'` -> **safe**
  - B: `'The wolf was safe and the sheep was dangerous, so the sheep was'` -> **dangerous**
- order 2: `'The sheep was safe and the wolf was dangerous, so the wolf was'` -> **dangerous**
  - B: `'The sheep was safe and the wolf was dangerous, so the sheep was'` -> **safe**
- order 2: `'The sheep was dangerous and the wolf was safe, so the wolf was'` -> **safe**
  - B: `'The sheep was dangerous and the wolf was safe, so the sheep was'` -> **dangerous**

## 35. ctrl|tea|ice|warm|cold (objects, ext)
- order 1: `'The tea was warm and the ice was cold, so the tea was'` -> **warm**
  - B: `'The tea was warm and the ice was cold, so the ice was'` -> **cold**
- order 1: `'The tea was cold and the ice was warm, so the tea was'` -> **cold**
  - B: `'The tea was cold and the ice was warm, so the ice was'` -> **warm**
- order 2: `'The ice was cold and the tea was warm, so the tea was'` -> **warm**
  - B: `'The ice was cold and the tea was warm, so the ice was'` -> **cold**
- order 2: `'The ice was warm and the tea was cold, so the tea was'` -> **cold**
  - B: `'The ice was warm and the tea was cold, so the ice was'` -> **warm**

## 36. ctrl|steel|cotton|hard|soft (objects, ext)
- order 1: `'The steel was hard and the cotton was soft, so the steel was'` -> **hard**
  - B: `'The steel was hard and the cotton was soft, so the cotton was'` -> **soft**
- order 1: `'The steel was soft and the cotton was hard, so the steel was'` -> **soft**
  - B: `'The steel was soft and the cotton was hard, so the cotton was'` -> **hard**
- order 2: `'The cotton was soft and the steel was hard, so the steel was'` -> **hard**
  - B: `'The cotton was soft and the steel was hard, so the cotton was'` -> **soft**
- order 2: `'The cotton was hard and the steel was soft, so the steel was'` -> **soft**
  - B: `'The cotton was hard and the steel was soft, so the cotton was'` -> **hard**

## 37. ctrl|palace|cabin|big|small (objects, ext)
- order 1: `'The palace was big and the cabin was small, so the palace was'` -> **big**
  - B: `'The palace was big and the cabin was small, so the cabin was'` -> **small**
- order 1: `'The palace was small and the cabin was big, so the palace was'` -> **small**
  - B: `'The palace was small and the cabin was big, so the cabin was'` -> **big**
- order 2: `'The cabin was small and the palace was big, so the palace was'` -> **big**
  - B: `'The cabin was small and the palace was big, so the cabin was'` -> **small**
- order 2: `'The cabin was big and the palace was small, so the palace was'` -> **small**
  - B: `'The cabin was big and the palace was small, so the cabin was'` -> **big**

## 38. ctrl|car|bike|fast|slow (objects, ext)
- order 1: `'The car was fast and the bike was slow, so the car was'` -> **fast**
  - B: `'The car was fast and the bike was slow, so the bike was'` -> **slow**
- order 1: `'The car was slow and the bike was fast, so the car was'` -> **slow**
  - B: `'The car was slow and the bike was fast, so the bike was'` -> **fast**
- order 2: `'The bike was slow and the car was fast, so the car was'` -> **fast**
  - B: `'The bike was slow and the car was fast, so the bike was'` -> **slow**
- order 2: `'The bike was fast and the car was slow, so the car was'` -> **slow**
  - B: `'The bike was fast and the car was slow, so the bike was'` -> **fast**

## 39. ctrl|ocean|stream|wide|narrow (objects, ext)
- order 1: `'The ocean was wide and the stream was narrow, so the ocean was'` -> **wide**
  - B: `'The ocean was wide and the stream was narrow, so the stream was'` -> **narrow**
- order 1: `'The ocean was narrow and the stream was wide, so the ocean was'` -> **narrow**
  - B: `'The ocean was narrow and the stream was wide, so the stream was'` -> **wide**
- order 2: `'The stream was narrow and the ocean was wide, so the ocean was'` -> **wide**
  - B: `'The stream was narrow and the ocean was wide, so the stream was'` -> **narrow**
- order 2: `'The stream was wide and the ocean was narrow, so the ocean was'` -> **narrow**
  - B: `'The stream was wide and the ocean was narrow, so the stream was'` -> **wide**

## 40. ctrl|castle|hut|big|small (objects, ext)
- order 1: `'The castle was big and the hut was small, so the castle was'` -> **big**
  - B: `'The castle was big and the hut was small, so the hut was'` -> **small**
- order 1: `'The castle was small and the hut was big, so the castle was'` -> **small**
  - B: `'The castle was small and the hut was big, so the hut was'` -> **big**
- order 2: `'The hut was small and the castle was big, so the castle was'` -> **big**
  - B: `'The hut was small and the castle was big, so the hut was'` -> **small**
- order 2: `'The hut was big and the castle was small, so the castle was'` -> **small**
  - B: `'The hut was big and the castle was small, so the hut was'` -> **big**

## 41. ctrl|iron|velvet|hard|soft (objects, ext)
- order 1: `'The iron was hard and the velvet was soft, so the iron was'` -> **hard**
  - B: `'The iron was hard and the velvet was soft, so the velvet was'` -> **soft**
- order 1: `'The iron was soft and the velvet was hard, so the iron was'` -> **soft**
  - B: `'The iron was soft and the velvet was hard, so the velvet was'` -> **hard**
- order 2: `'The velvet was soft and the iron was hard, so the iron was'` -> **hard**
  - B: `'The velvet was soft and the iron was hard, so the velvet was'` -> **soft**
- order 2: `'The velvet was hard and the iron was soft, so the iron was'` -> **soft**
  - B: `'The velvet was hard and the iron was soft, so the velvet was'` -> **hard**

## 42. ctrl|lake|dust|wet|dry (objects, ext)
- order 1: `'The lake was wet and the dust was dry, so the lake was'` -> **wet**
  - B: `'The lake was wet and the dust was dry, so the dust was'` -> **dry**
- order 1: `'The lake was dry and the dust was wet, so the lake was'` -> **dry**
  - B: `'The lake was dry and the dust was wet, so the dust was'` -> **wet**
- order 2: `'The dust was dry and the lake was wet, so the lake was'` -> **wet**
  - B: `'The dust was dry and the lake was wet, so the dust was'` -> **dry**
- order 2: `'The dust was wet and the lake was dry, so the lake was'` -> **dry**
  - B: `'The dust was wet and the lake was dry, so the dust was'` -> **wet**

## 43. ctrl|horse|ant|big|small (objects, ext)
- order 1: `'The horse was big and the ant was small, so the horse was'` -> **big**
  - B: `'The horse was big and the ant was small, so the ant was'` -> **small**
- order 1: `'The horse was small and the ant was big, so the horse was'` -> **small**
  - B: `'The horse was small and the ant was big, so the ant was'` -> **big**
- order 2: `'The ant was small and the horse was big, so the horse was'` -> **big**
  - B: `'The ant was small and the horse was big, so the ant was'` -> **small**
- order 2: `'The ant was big and the horse was small, so the horse was'` -> **small**
  - B: `'The ant was big and the horse was small, so the ant was'` -> **big**

## 44. ctrl|ocean|desert|wet|dry (objects, ext)
- order 1: `'The ocean was wet and the desert was dry, so the ocean was'` -> **wet**
  - B: `'The ocean was wet and the desert was dry, so the desert was'` -> **dry**
- order 1: `'The ocean was dry and the desert was wet, so the ocean was'` -> **dry**
  - B: `'The ocean was dry and the desert was wet, so the desert was'` -> **wet**
- order 2: `'The desert was dry and the ocean was wet, so the ocean was'` -> **wet**
  - B: `'The desert was dry and the ocean was wet, so the desert was'` -> **dry**
- order 2: `'The desert was wet and the ocean was dry, so the ocean was'` -> **dry**
  - B: `'The desert was wet and the ocean was dry, so the desert was'` -> **wet**

## 45. ctrl|stadium|graveyard|full|empty (objects, ext)
- order 1: `'The stadium was full and the graveyard was empty, so the stadium was'` -> **full**
  - B: `'The stadium was full and the graveyard was empty, so the graveyard was'` -> **empty**
- order 1: `'The stadium was empty and the graveyard was full, so the stadium was'` -> **empty**
  - B: `'The stadium was empty and the graveyard was full, so the graveyard was'` -> **full**
- order 2: `'The graveyard was empty and the stadium was full, so the stadium was'` -> **full**
  - B: `'The graveyard was empty and the stadium was full, so the graveyard was'` -> **empty**
- order 2: `'The graveyard was full and the stadium was empty, so the stadium was'` -> **empty**
  - B: `'The graveyard was full and the stadium was empty, so the graveyard was'` -> **full**

## 46. ctrl|road|path|wide|narrow (objects, ext)
- order 1: `'The road was wide and the path was narrow, so the road was'` -> **wide**
  - B: `'The road was wide and the path was narrow, so the path was'` -> **narrow**
- order 1: `'The road was narrow and the path was wide, so the road was'` -> **narrow**
  - B: `'The road was narrow and the path was wide, so the path was'` -> **wide**
- order 2: `'The path was narrow and the road was wide, so the road was'` -> **wide**
  - B: `'The path was narrow and the road was wide, so the path was'` -> **narrow**
- order 2: `'The path was wide and the road was narrow, so the road was'` -> **narrow**
  - B: `'The path was wide and the road was narrow, so the path was'` -> **wide**

## 47. ctrl|train|wagon|fast|slow (objects, ext)
- order 1: `'The train was fast and the wagon was slow, so the train was'` -> **fast**
  - B: `'The train was fast and the wagon was slow, so the wagon was'` -> **slow**
- order 1: `'The train was slow and the wagon was fast, so the train was'` -> **slow**
  - B: `'The train was slow and the wagon was fast, so the wagon was'` -> **fast**
- order 2: `'The wagon was slow and the train was fast, so the train was'` -> **fast**
  - B: `'The wagon was slow and the train was fast, so the wagon was'` -> **slow**
- order 2: `'The wagon was fast and the train was slow, so the train was'` -> **slow**
  - B: `'The wagon was fast and the train was slow, so the wagon was'` -> **fast**

## 48. 3HYV4299H2A2K9M563ZGO948PS98ER (objects, core)
- order 1: `'The head was bald and the crowd was vast, so the head was'` -> **bald**
  - B: `'The head was bald and the crowd was vast, so the crowd was'` -> **vast**
- order 2: `'The crowd was vast and the head was bald, so the head was'` -> **bald**
  - B: `'The crowd was vast and the head was bald, so the crowd was'` -> **vast**

## 49. ctrl|airplane|bee|large|tiny (objects, ext)
- order 1: `'The airplane was large and the bee was tiny, so the airplane was'` -> **large**
  - B: `'The airplane was large and the bee was tiny, so the bee was'` -> **tiny**
- order 1: `'The airplane was tiny and the bee was large, so the airplane was'` -> **tiny**
  - B: `'The airplane was tiny and the bee was large, so the bee was'` -> **large**
- order 2: `'The bee was tiny and the airplane was large, so the airplane was'` -> **large**
  - B: `'The bee was tiny and the airplane was large, so the bee was'` -> **tiny**
- order 2: `'The bee was large and the airplane was tiny, so the airplane was'` -> **tiny**
  - B: `'The bee was large and the airplane was tiny, so the bee was'` -> **large**

## 50. ctrl|highway|graveyard|noisy|silent (objects, ext)
- order 1: `'The highway was noisy and the graveyard was silent, so the highway was'` -> **noisy**
  - B: `'The highway was noisy and the graveyard was silent, so the graveyard was'` -> **silent**
- order 1: `'The highway was silent and the graveyard was noisy, so the highway was'` -> **silent**
  - B: `'The highway was silent and the graveyard was noisy, so the graveyard was'` -> **noisy**
- order 2: `'The graveyard was silent and the highway was noisy, so the highway was'` -> **noisy**
  - B: `'The graveyard was silent and the highway was noisy, so the graveyard was'` -> **silent**
- order 2: `'The graveyard was noisy and the highway was silent, so the highway was'` -> **silent**
  - B: `'The graveyard was noisy and the highway was silent, so the graveyard was'` -> **noisy**
