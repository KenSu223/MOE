# l_bind: 50 seeded samples (qwen-family texts; random_state 0)

## 1. l_bind|watch|apple|white|gold (synthetic)
- order 1: `'The watch is in the white box and the apple is in the gold box. The white box contains the'` -> **watch** | B: `'The watch is in the white box and the apple is in the gold box. The gold box contains the'` -> **apple**
- order 2: `'The apple is in the gold box and the watch is in the white box. The white box contains the'` -> **watch** | B: `'The apple is in the gold box and the watch is in the white box. The gold box contains the'` -> **apple**

## 2. l_bind|stone|egg|brown|white (synthetic)
- order 1: `'The stone is in the brown box and the egg is in the white box. The brown box contains the'` -> **stone** | B: `'The stone is in the brown box and the egg is in the white box. The white box contains the'` -> **egg**
- order 2: `'The egg is in the white box and the stone is in the brown box. The brown box contains the'` -> **stone** | B: `'The egg is in the white box and the stone is in the brown box. The white box contains the'` -> **egg**

## 3. l_bind|book|plate|black|gold (synthetic)
- order 1: `'The book is in the black box and the plate is in the gold box. The black box contains the'` -> **book** | B: `'The book is in the black box and the plate is in the gold box. The gold box contains the'` -> **plate**
- order 2: `'The plate is in the gold box and the book is in the black box. The black box contains the'` -> **book** | B: `'The plate is in the gold box and the book is in the black box. The gold box contains the'` -> **plate**

## 4. l_bind|flower|bottle|gray|green (synthetic)
- order 1: `'The flower is in the gray box and the bottle is in the green box. The gray box contains the'` -> **flower** | B: `'The flower is in the gray box and the bottle is in the green box. The green box contains the'` -> **bottle**
- order 2: `'The bottle is in the green box and the flower is in the gray box. The gray box contains the'` -> **flower** | B: `'The bottle is in the green box and the flower is in the gray box. The green box contains the'` -> **bottle**

## 5. l_bind|coin|lemon|green|silver (synthetic)
- order 1: `'The coin is in the green box and the lemon is in the silver box. The green box contains the'` -> **coin** | B: `'The coin is in the green box and the lemon is in the silver box. The silver box contains the'` -> **lemon**
- order 2: `'The lemon is in the silver box and the coin is in the green box. The green box contains the'` -> **coin** | B: `'The lemon is in the silver box and the coin is in the green box. The silver box contains the'` -> **lemon**

## 6. l_bind|bell|spoon|yellow|white (synthetic)
- order 1: `'The bell is in the yellow box and the spoon is in the white box. The yellow box contains the'` -> **bell** | B: `'The bell is in the yellow box and the spoon is in the white box. The white box contains the'` -> **spoon**
- order 2: `'The spoon is in the white box and the bell is in the yellow box. The yellow box contains the'` -> **bell** | B: `'The spoon is in the white box and the bell is in the yellow box. The white box contains the'` -> **spoon**

## 7. l_bind|book|clock|black|silver (synthetic)
- order 1: `'The book is in the black box and the clock is in the silver box. The black box contains the'` -> **book** | B: `'The book is in the black box and the clock is in the silver box. The silver box contains the'` -> **clock**
- order 2: `'The clock is in the silver box and the book is in the black box. The black box contains the'` -> **book** | B: `'The clock is in the silver box and the book is in the black box. The silver box contains the'` -> **clock**

## 8. l_bind|phone|fork|purple|brown (synthetic)
- order 1: `'The phone is in the purple box and the fork is in the brown box. The purple box contains the'` -> **phone** | B: `'The phone is in the purple box and the fork is in the brown box. The brown box contains the'` -> **fork**
- order 2: `'The fork is in the brown box and the phone is in the purple box. The purple box contains the'` -> **phone** | B: `'The fork is in the brown box and the phone is in the purple box. The brown box contains the'` -> **fork**

## 9. l_bind|bottle|shoe|pink|green (synthetic)
- order 1: `'The bottle is in the pink box and the shoe is in the green box. The pink box contains the'` -> **bottle** | B: `'The bottle is in the pink box and the shoe is in the green box. The green box contains the'` -> **shoe**
- order 2: `'The shoe is in the green box and the bottle is in the pink box. The pink box contains the'` -> **bottle** | B: `'The shoe is in the green box and the bottle is in the pink box. The green box contains the'` -> **shoe**

## 10. l_bind|book|card|orange|brown (synthetic)
- order 1: `'The book is in the orange box and the card is in the brown box. The orange box contains the'` -> **book** | B: `'The book is in the orange box and the card is in the brown box. The brown box contains the'` -> **card**
- order 2: `'The card is in the brown box and the book is in the orange box. The orange box contains the'` -> **book** | B: `'The card is in the brown box and the book is in the orange box. The brown box contains the'` -> **card**

## 11. l_bind|wallet|ring|blue|gray (synthetic)
- order 1: `'The wallet is in the blue box and the ring is in the gray box. The blue box contains the'` -> **wallet** | B: `'The wallet is in the blue box and the ring is in the gray box. The gray box contains the'` -> **ring**
- order 2: `'The ring is in the gray box and the wallet is in the blue box. The blue box contains the'` -> **wallet** | B: `'The ring is in the gray box and the wallet is in the blue box. The gray box contains the'` -> **ring**

## 12. l_bind|spoon|hat|pink|blue (synthetic)
- order 1: `'The spoon is in the pink box and the hat is in the blue box. The pink box contains the'` -> **spoon** | B: `'The spoon is in the pink box and the hat is in the blue box. The blue box contains the'` -> **hat**
- order 2: `'The hat is in the blue box and the spoon is in the pink box. The pink box contains the'` -> **spoon** | B: `'The hat is in the blue box and the spoon is in the pink box. The blue box contains the'` -> **hat**

## 13. l_bind|spoon|cookie|red|blue (synthetic)
- order 1: `'The spoon is in the red box and the cookie is in the blue box. The red box contains the'` -> **spoon** | B: `'The spoon is in the red box and the cookie is in the blue box. The blue box contains the'` -> **cookie**
- order 2: `'The cookie is in the blue box and the spoon is in the red box. The red box contains the'` -> **spoon** | B: `'The cookie is in the blue box and the spoon is in the red box. The blue box contains the'` -> **cookie**

## 14. l_bind|shoe|button|gold|silver (synthetic)
- order 1: `'The shoe is in the gold box and the button is in the silver box. The gold box contains the'` -> **shoe** | B: `'The shoe is in the gold box and the button is in the silver box. The silver box contains the'` -> **button**
- order 2: `'The button is in the silver box and the shoe is in the gold box. The gold box contains the'` -> **shoe** | B: `'The button is in the silver box and the shoe is in the gold box. The silver box contains the'` -> **button**

## 15. l_bind|doll|phone|pink|yellow (synthetic)
- order 1: `'The doll is in the pink box and the phone is in the yellow box. The pink box contains the'` -> **doll** | B: `'The doll is in the pink box and the phone is in the yellow box. The yellow box contains the'` -> **phone**
- order 2: `'The phone is in the yellow box and the doll is in the pink box. The pink box contains the'` -> **doll** | B: `'The phone is in the yellow box and the doll is in the pink box. The yellow box contains the'` -> **phone**

## 16. l_bind|stone|letter|black|blue (synthetic)
- order 1: `'The stone is in the black box and the letter is in the blue box. The black box contains the'` -> **stone** | B: `'The stone is in the black box and the letter is in the blue box. The blue box contains the'` -> **letter**
- order 2: `'The letter is in the blue box and the stone is in the black box. The black box contains the'` -> **stone** | B: `'The letter is in the blue box and the stone is in the black box. The blue box contains the'` -> **letter**

## 17. l_bind|brush|apple|pink|yellow (synthetic)
- order 1: `'The brush is in the pink box and the apple is in the yellow box. The pink box contains the'` -> **brush** | B: `'The brush is in the pink box and the apple is in the yellow box. The yellow box contains the'` -> **apple**
- order 2: `'The apple is in the yellow box and the brush is in the pink box. The pink box contains the'` -> **brush** | B: `'The apple is in the yellow box and the brush is in the pink box. The yellow box contains the'` -> **apple**

## 18. l_bind|toy|brush|white|gray (synthetic)
- order 1: `'The toy is in the white box and the brush is in the gray box. The white box contains the'` -> **toy** | B: `'The toy is in the white box and the brush is in the gray box. The gray box contains the'` -> **brush**
- order 2: `'The brush is in the gray box and the toy is in the white box. The white box contains the'` -> **toy** | B: `'The brush is in the gray box and the toy is in the white box. The gray box contains the'` -> **brush**

## 19. l_bind|battery|bottle|yellow|brown (synthetic)
- order 1: `'The battery is in the yellow box and the bottle is in the brown box. The yellow box contains the'` -> **battery** | B: `'The battery is in the yellow box and the bottle is in the brown box. The brown box contains the'` -> **bottle**
- order 2: `'The bottle is in the brown box and the battery is in the yellow box. The yellow box contains the'` -> **battery** | B: `'The bottle is in the brown box and the battery is in the yellow box. The brown box contains the'` -> **bottle**

## 20. l_bind|pen|sock|green|gold (synthetic)
- order 1: `'The pen is in the green box and the sock is in the gold box. The green box contains the'` -> **pen** | B: `'The pen is in the green box and the sock is in the gold box. The gold box contains the'` -> **sock**
- order 2: `'The sock is in the gold box and the pen is in the green box. The green box contains the'` -> **pen** | B: `'The sock is in the gold box and the pen is in the green box. The gold box contains the'` -> **sock**

## 21. l_bind|cookie|button|red|gold (synthetic)
- order 1: `'The cookie is in the red box and the button is in the gold box. The red box contains the'` -> **cookie** | B: `'The cookie is in the red box and the button is in the gold box. The gold box contains the'` -> **button**
- order 2: `'The button is in the gold box and the cookie is in the red box. The red box contains the'` -> **cookie** | B: `'The button is in the gold box and the cookie is in the red box. The gold box contains the'` -> **button**

## 22. l_bind|watch|bell|blue|orange (synthetic)
- order 1: `'The watch is in the blue box and the bell is in the orange box. The blue box contains the'` -> **watch** | B: `'The watch is in the blue box and the bell is in the orange box. The orange box contains the'` -> **bell**
- order 2: `'The bell is in the orange box and the watch is in the blue box. The blue box contains the'` -> **watch** | B: `'The bell is in the orange box and the watch is in the blue box. The orange box contains the'` -> **bell**

## 23. l_bind|mirror|cup|yellow|black (synthetic)
- order 1: `'The mirror is in the yellow box and the cup is in the black box. The yellow box contains the'` -> **mirror** | B: `'The mirror is in the yellow box and the cup is in the black box. The black box contains the'` -> **cup**
- order 2: `'The cup is in the black box and the mirror is in the yellow box. The yellow box contains the'` -> **mirror** | B: `'The cup is in the black box and the mirror is in the yellow box. The black box contains the'` -> **cup**

## 24. l_bind|coin|mirror|orange|blue (synthetic)
- order 1: `'The coin is in the orange box and the mirror is in the blue box. The orange box contains the'` -> **coin** | B: `'The coin is in the orange box and the mirror is in the blue box. The blue box contains the'` -> **mirror**
- order 2: `'The mirror is in the blue box and the coin is in the orange box. The orange box contains the'` -> **coin** | B: `'The mirror is in the blue box and the coin is in the orange box. The blue box contains the'` -> **mirror**

## 25. l_bind|bottle|egg|pink|red (synthetic)
- order 1: `'The bottle is in the pink box and the egg is in the red box. The pink box contains the'` -> **bottle** | B: `'The bottle is in the pink box and the egg is in the red box. The red box contains the'` -> **egg**
- order 2: `'The egg is in the red box and the bottle is in the pink box. The pink box contains the'` -> **bottle** | B: `'The egg is in the red box and the bottle is in the pink box. The red box contains the'` -> **egg**

## 26. l_bind|stone|fork|gold|blue (synthetic)
- order 1: `'The stone is in the gold box and the fork is in the blue box. The gold box contains the'` -> **stone** | B: `'The stone is in the gold box and the fork is in the blue box. The blue box contains the'` -> **fork**
- order 2: `'The fork is in the blue box and the stone is in the gold box. The gold box contains the'` -> **stone** | B: `'The fork is in the blue box and the stone is in the gold box. The blue box contains the'` -> **fork**

## 27. l_bind|bell|phone|gold|gray (synthetic)
- order 1: `'The bell is in the gold box and the phone is in the gray box. The gold box contains the'` -> **bell** | B: `'The bell is in the gold box and the phone is in the gray box. The gray box contains the'` -> **phone**
- order 2: `'The phone is in the gray box and the bell is in the gold box. The gold box contains the'` -> **bell** | B: `'The phone is in the gray box and the bell is in the gold box. The gray box contains the'` -> **phone**

## 28. l_bind|egg|button|yellow|gold (synthetic)
- order 1: `'The egg is in the yellow box and the button is in the gold box. The yellow box contains the'` -> **egg** | B: `'The egg is in the yellow box and the button is in the gold box. The gold box contains the'` -> **button**
- order 2: `'The button is in the gold box and the egg is in the yellow box. The yellow box contains the'` -> **egg** | B: `'The button is in the gold box and the egg is in the yellow box. The gold box contains the'` -> **button**

## 29. l_bind|book|ticket|gray|orange (synthetic)
- order 1: `'The book is in the gray box and the ticket is in the orange box. The gray box contains the'` -> **book** | B: `'The book is in the gray box and the ticket is in the orange box. The orange box contains the'` -> **ticket**
- order 2: `'The ticket is in the orange box and the book is in the gray box. The gray box contains the'` -> **book** | B: `'The ticket is in the orange box and the book is in the gray box. The orange box contains the'` -> **ticket**

## 30. l_bind|key|shell|purple|brown (synthetic)
- order 1: `'The key is in the purple box and the shell is in the brown box. The purple box contains the'` -> **key** | B: `'The key is in the purple box and the shell is in the brown box. The brown box contains the'` -> **shell**
- order 2: `'The shell is in the brown box and the key is in the purple box. The purple box contains the'` -> **key** | B: `'The shell is in the brown box and the key is in the purple box. The brown box contains the'` -> **shell**

## 31. l_bind|marble|rope|gold|orange (synthetic)
- order 1: `'The marble is in the gold box and the rope is in the orange box. The gold box contains the'` -> **marble** | B: `'The marble is in the gold box and the rope is in the orange box. The orange box contains the'` -> **rope**
- order 2: `'The rope is in the orange box and the marble is in the gold box. The gold box contains the'` -> **marble** | B: `'The rope is in the orange box and the marble is in the gold box. The orange box contains the'` -> **rope**

## 32. l_bind|ball|bottle|orange|yellow (synthetic)
- order 1: `'The ball is in the orange box and the bottle is in the yellow box. The orange box contains the'` -> **ball** | B: `'The ball is in the orange box and the bottle is in the yellow box. The yellow box contains the'` -> **bottle**
- order 2: `'The bottle is in the yellow box and the ball is in the orange box. The orange box contains the'` -> **ball** | B: `'The bottle is in the yellow box and the ball is in the orange box. The yellow box contains the'` -> **bottle**

## 33. l_bind|candle|plate|yellow|purple (synthetic)
- order 1: `'The candle is in the yellow box and the plate is in the purple box. The yellow box contains the'` -> **candle** | B: `'The candle is in the yellow box and the plate is in the purple box. The purple box contains the'` -> **plate**
- order 2: `'The plate is in the purple box and the candle is in the yellow box. The yellow box contains the'` -> **candle** | B: `'The plate is in the purple box and the candle is in the yellow box. The purple box contains the'` -> **plate**

## 34. l_bind|book|coin|yellow|purple (synthetic)
- order 1: `'The book is in the yellow box and the coin is in the purple box. The yellow box contains the'` -> **book** | B: `'The book is in the yellow box and the coin is in the purple box. The purple box contains the'` -> **coin**
- order 2: `'The coin is in the purple box and the book is in the yellow box. The yellow box contains the'` -> **book** | B: `'The coin is in the purple box and the book is in the yellow box. The purple box contains the'` -> **coin**

## 35. l_bind|coin|wallet|black|silver (synthetic)
- order 1: `'The coin is in the black box and the wallet is in the silver box. The black box contains the'` -> **coin** | B: `'The coin is in the black box and the wallet is in the silver box. The silver box contains the'` -> **wallet**
- order 2: `'The wallet is in the silver box and the coin is in the black box. The black box contains the'` -> **coin** | B: `'The wallet is in the silver box and the coin is in the black box. The silver box contains the'` -> **wallet**

## 36. l_bind|apple|ring|gray|orange (synthetic)
- order 1: `'The apple is in the gray box and the ring is in the orange box. The gray box contains the'` -> **apple** | B: `'The apple is in the gray box and the ring is in the orange box. The orange box contains the'` -> **ring**
- order 2: `'The ring is in the orange box and the apple is in the gray box. The gray box contains the'` -> **apple** | B: `'The ring is in the orange box and the apple is in the gray box. The orange box contains the'` -> **ring**

## 37. l_bind|key|phone|black|blue (synthetic)
- order 1: `'The key is in the black box and the phone is in the blue box. The black box contains the'` -> **key** | B: `'The key is in the black box and the phone is in the blue box. The blue box contains the'` -> **phone**
- order 2: `'The phone is in the blue box and the key is in the black box. The black box contains the'` -> **key** | B: `'The phone is in the blue box and the key is in the black box. The blue box contains the'` -> **phone**

## 38. l_bind|cookie|towel|gold|purple (synthetic)
- order 1: `'The cookie is in the gold box and the towel is in the purple box. The gold box contains the'` -> **cookie** | B: `'The cookie is in the gold box and the towel is in the purple box. The purple box contains the'` -> **towel**
- order 2: `'The towel is in the purple box and the cookie is in the gold box. The gold box contains the'` -> **cookie** | B: `'The towel is in the purple box and the cookie is in the gold box. The purple box contains the'` -> **towel**

## 39. l_bind|coin|lemon|green|blue (synthetic)
- order 1: `'The coin is in the green box and the lemon is in the blue box. The green box contains the'` -> **coin** | B: `'The coin is in the green box and the lemon is in the blue box. The blue box contains the'` -> **lemon**
- order 2: `'The lemon is in the blue box and the coin is in the green box. The green box contains the'` -> **coin** | B: `'The lemon is in the blue box and the coin is in the green box. The blue box contains the'` -> **lemon**

## 40. l_bind|knife|bottle|brown|purple (synthetic)
- order 1: `'The knife is in the brown box and the bottle is in the purple box. The brown box contains the'` -> **knife** | B: `'The knife is in the brown box and the bottle is in the purple box. The purple box contains the'` -> **bottle**
- order 2: `'The bottle is in the purple box and the knife is in the brown box. The brown box contains the'` -> **knife** | B: `'The bottle is in the purple box and the knife is in the brown box. The purple box contains the'` -> **bottle**

## 41. l_bind|phone|towel|purple|red (synthetic)
- order 1: `'The phone is in the purple box and the towel is in the red box. The purple box contains the'` -> **phone** | B: `'The phone is in the purple box and the towel is in the red box. The red box contains the'` -> **towel**
- order 2: `'The towel is in the red box and the phone is in the purple box. The purple box contains the'` -> **phone** | B: `'The towel is in the red box and the phone is in the purple box. The red box contains the'` -> **towel**

## 42. l_bind|comb|mirror|green|pink (synthetic)
- order 1: `'The comb is in the green box and the mirror is in the pink box. The green box contains the'` -> **comb** | B: `'The comb is in the green box and the mirror is in the pink box. The pink box contains the'` -> **mirror**
- order 2: `'The mirror is in the pink box and the comb is in the green box. The green box contains the'` -> **comb** | B: `'The mirror is in the pink box and the comb is in the green box. The pink box contains the'` -> **mirror**

## 43. l_bind|candle|spoon|gold|orange (synthetic)
- order 1: `'The candle is in the gold box and the spoon is in the orange box. The gold box contains the'` -> **candle** | B: `'The candle is in the gold box and the spoon is in the orange box. The orange box contains the'` -> **spoon**
- order 2: `'The spoon is in the orange box and the candle is in the gold box. The gold box contains the'` -> **candle** | B: `'The spoon is in the orange box and the candle is in the gold box. The orange box contains the'` -> **spoon**

## 44. l_bind|shell|shoe|white|green (synthetic)
- order 1: `'The shell is in the white box and the shoe is in the green box. The white box contains the'` -> **shell** | B: `'The shell is in the white box and the shoe is in the green box. The green box contains the'` -> **shoe**
- order 2: `'The shoe is in the green box and the shell is in the white box. The white box contains the'` -> **shell** | B: `'The shoe is in the green box and the shell is in the white box. The green box contains the'` -> **shoe**

## 45. l_bind|stone|shoe|red|black (synthetic)
- order 1: `'The stone is in the red box and the shoe is in the black box. The red box contains the'` -> **stone** | B: `'The stone is in the red box and the shoe is in the black box. The black box contains the'` -> **shoe**
- order 2: `'The shoe is in the black box and the stone is in the red box. The red box contains the'` -> **stone** | B: `'The shoe is in the black box and the stone is in the red box. The black box contains the'` -> **shoe**

## 46. l_bind|sock|candle|yellow|silver (synthetic)
- order 1: `'The sock is in the yellow box and the candle is in the silver box. The yellow box contains the'` -> **sock** | B: `'The sock is in the yellow box and the candle is in the silver box. The silver box contains the'` -> **candle**
- order 2: `'The candle is in the silver box and the sock is in the yellow box. The yellow box contains the'` -> **sock** | B: `'The candle is in the silver box and the sock is in the yellow box. The silver box contains the'` -> **candle**

## 47. l_bind|stone|wallet|red|black (synthetic)
- order 1: `'The stone is in the red box and the wallet is in the black box. The red box contains the'` -> **stone** | B: `'The stone is in the red box and the wallet is in the black box. The black box contains the'` -> **wallet**
- order 2: `'The wallet is in the black box and the stone is in the red box. The red box contains the'` -> **stone** | B: `'The wallet is in the black box and the stone is in the red box. The black box contains the'` -> **wallet**

## 48. l_bind|book|marble|pink|orange (synthetic)
- order 1: `'The book is in the pink box and the marble is in the orange box. The pink box contains the'` -> **book** | B: `'The book is in the pink box and the marble is in the orange box. The orange box contains the'` -> **marble**
- order 2: `'The marble is in the orange box and the book is in the pink box. The pink box contains the'` -> **book** | B: `'The marble is in the orange box and the book is in the pink box. The orange box contains the'` -> **marble**

## 49. l_bind|spoon|knife|white|gold (synthetic)
- order 1: `'The spoon is in the white box and the knife is in the gold box. The white box contains the'` -> **spoon** | B: `'The spoon is in the white box and the knife is in the gold box. The gold box contains the'` -> **knife**
- order 2: `'The knife is in the gold box and the spoon is in the white box. The white box contains the'` -> **spoon** | B: `'The knife is in the gold box and the spoon is in the white box. The gold box contains the'` -> **knife**

## 50. l_bind|map|brush|blue|white (synthetic)
- order 1: `'The map is in the blue box and the brush is in the white box. The blue box contains the'` -> **map** | B: `'The map is in the blue box and the brush is in the white box. The white box contains the'` -> **brush**
- order 2: `'The brush is in the white box and the map is in the blue box. The blue box contains the'` -> **map** | B: `'The brush is in the white box and the map is in the blue box. The white box contains the'` -> **brush**
