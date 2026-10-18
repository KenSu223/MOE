# l_ind: 50 seeded samples (qwen-family texts; random_state 0)

## 1. l_ind|wallet|cloud|tunnel|table|bridge|kitchen|camera|island (synthetic)
- order 1: `'\nwallet: tunnel\ncloud: table\nbridge: kitchen\ncamera: island\nwallet:'` -> **tunnel** | B: `'\nwallet: tunnel\ncloud: table\nbridge: kitchen\ncamera: island\ncloud:'` -> **table**
- order 2: `'\ncloud: table\nwallet: tunnel\nbridge: kitchen\ncamera: island\nwallet:'` -> **tunnel** | B: `'\ncloud: table\nwallet: tunnel\nbridge: kitchen\ncamera: island\ncloud:'` -> **table**

## 2. l_ind|paper|castle|candle|carpet|circle|helmet|anchor|dog (synthetic)
- order 1: `'\npaper: candle\ncastle: carpet\ncircle: helmet\nanchor: dog\npaper:'` -> **candle** | B: `'\npaper: candle\ncastle: carpet\ncircle: helmet\nanchor: dog\ncastle:'` -> **carpet**
- order 2: `'\ncastle: carpet\npaper: candle\ncircle: helmet\nanchor: dog\npaper:'` -> **candle** | B: `'\ncastle: carpet\npaper: candle\ncircle: helmet\nanchor: dog\ncastle:'` -> **carpet**

## 3. l_ind|castle|dog|guitar|bread|cup|sugar|paper|hammer (synthetic)
- order 1: `'\ncastle: guitar\ndog: bread\ncup: sugar\npaper: hammer\ncastle:'` -> **guitar** | B: `'\ncastle: guitar\ndog: bread\ncup: sugar\npaper: hammer\ndog:'` -> **bread**
- order 2: `'\ndog: bread\ncastle: guitar\ncup: sugar\npaper: hammer\ncastle:'` -> **guitar** | B: `'\ndog: bread\ncastle: guitar\ncup: sugar\npaper: hammer\ndog:'` -> **bread**

## 4. l_ind|river|engine|needle|horse|bucket|bread|bridge|paper (synthetic)
- order 1: `'\nriver: needle\nengine: horse\nbucket: bread\nbridge: paper\nriver:'` -> **needle** | B: `'\nriver: needle\nengine: horse\nbucket: bread\nbridge: paper\nengine:'` -> **horse**
- order 2: `'\nengine: horse\nriver: needle\nbucket: bread\nbridge: paper\nriver:'` -> **needle** | B: `'\nengine: horse\nriver: needle\nbucket: bread\nbridge: paper\nengine:'` -> **horse**

## 5. l_ind|window|cloud|farmer|guitar|castle|oven|table|forest (synthetic)
- order 1: `'\nwindow: farmer\ncloud: guitar\ncastle: oven\ntable: forest\nwindow:'` -> **farmer** | B: `'\nwindow: farmer\ncloud: guitar\ncastle: oven\ntable: forest\ncloud:'` -> **guitar**
- order 2: `'\ncloud: guitar\nwindow: farmer\ncastle: oven\ntable: forest\nwindow:'` -> **farmer** | B: `'\ncloud: guitar\nwindow: farmer\ncastle: oven\ntable: forest\ncloud:'` -> **guitar**

## 6. l_ind|oven|river|orange|ocean|bread|sugar|bucket|winter (synthetic)
- order 1: `'\noven: orange\nriver: ocean\nbread: sugar\nbucket: winter\noven:'` -> **orange** | B: `'\noven: orange\nriver: ocean\nbread: sugar\nbucket: winter\nriver:'` -> **ocean**
- order 2: `'\nriver: ocean\noven: orange\nbread: sugar\nbucket: winter\noven:'` -> **orange** | B: `'\nriver: ocean\noven: orange\nbread: sugar\nbucket: winter\nriver:'` -> **ocean**

## 7. l_ind|castle|wallet|ocean|circle|letter|valley|oven|window (synthetic)
- order 1: `'\ncastle: ocean\nwallet: circle\nletter: valley\noven: window\ncastle:'` -> **ocean** | B: `'\ncastle: ocean\nwallet: circle\nletter: valley\noven: window\nwallet:'` -> **circle**
- order 2: `'\nwallet: circle\ncastle: ocean\nletter: valley\noven: window\ncastle:'` -> **ocean** | B: `'\nwallet: circle\ncastle: ocean\nletter: valley\noven: window\nwallet:'` -> **circle**

## 8. l_ind|castle|engine|desert|window|bucket|needle|camera|winter (synthetic)
- order 1: `'\ncastle: desert\nengine: window\nbucket: needle\ncamera: winter\ncastle:'` -> **desert** | B: `'\ncastle: desert\nengine: window\nbucket: needle\ncamera: winter\nengine:'` -> **window**
- order 2: `'\nengine: window\ncastle: desert\nbucket: needle\ncamera: winter\ncastle:'` -> **desert** | B: `'\nengine: window\ncastle: desert\nbucket: needle\ncamera: winter\nengine:'` -> **window**

## 9. l_ind|letter|camera|village|ocean|button|glass|school|jacket (synthetic)
- order 1: `'\nletter: village\ncamera: ocean\nbutton: glass\nschool: jacket\nletter:'` -> **village** | B: `'\nletter: village\ncamera: ocean\nbutton: glass\nschool: jacket\ncamera:'` -> **ocean**
- order 2: `'\ncamera: ocean\nletter: village\nbutton: glass\nschool: jacket\nletter:'` -> **village** | B: `'\ncamera: ocean\nletter: village\nbutton: glass\nschool: jacket\ncamera:'` -> **ocean**

## 10. l_ind|cup|camera|mirror|valley|window|desert|oven|wallet (synthetic)
- order 1: `'\ncup: mirror\ncamera: valley\nwindow: desert\noven: wallet\ncup:'` -> **mirror** | B: `'\ncup: mirror\ncamera: valley\nwindow: desert\noven: wallet\ncamera:'` -> **valley**
- order 2: `'\ncamera: valley\ncup: mirror\nwindow: desert\noven: wallet\ncup:'` -> **mirror** | B: `'\ncamera: valley\ncup: mirror\nwindow: desert\noven: wallet\ncamera:'` -> **valley**

## 11. l_ind|dog|window|mountain|table|cloud|helmet|bridge|desert (synthetic)
- order 1: `'\ndog: mountain\nwindow: table\ncloud: helmet\nbridge: desert\ndog:'` -> **mountain** | B: `'\ndog: mountain\nwindow: table\ncloud: helmet\nbridge: desert\nwindow:'` -> **table**
- order 2: `'\nwindow: table\ndog: mountain\ncloud: helmet\nbridge: desert\ndog:'` -> **mountain** | B: `'\nwindow: table\ndog: mountain\ncloud: helmet\nbridge: desert\nwindow:'` -> **table**

## 12. l_ind|school|table|saddle|farmer|bridge|tunnel|cup|glass (synthetic)
- order 1: `'\nschool: saddle\ntable: farmer\nbridge: tunnel\ncup: glass\nschool:'` -> **saddle** | B: `'\nschool: saddle\ntable: farmer\nbridge: tunnel\ncup: glass\ntable:'` -> **farmer**
- order 2: `'\ntable: farmer\nschool: saddle\nbridge: tunnel\ncup: glass\nschool:'` -> **saddle** | B: `'\ntable: farmer\nschool: saddle\nbridge: tunnel\ncup: glass\ntable:'` -> **farmer**

## 13. l_ind|button|river|cabin|basket|table|forest|cloud|yellow (synthetic)
- order 1: `'\nbutton: cabin\nriver: basket\ntable: forest\ncloud: yellow\nbutton:'` -> **cabin** | B: `'\nbutton: cabin\nriver: basket\ntable: forest\ncloud: yellow\nriver:'` -> **basket**
- order 2: `'\nriver: basket\nbutton: cabin\ntable: forest\ncloud: yellow\nbutton:'` -> **cabin** | B: `'\nriver: basket\nbutton: cabin\ntable: forest\ncloud: yellow\nriver:'` -> **basket**

## 14. l_ind|cup|camera|chair|queen|anchor|horse|bucket|teacher (synthetic)
- order 1: `'\ncup: chair\ncamera: queen\nanchor: horse\nbucket: teacher\ncup:'` -> **chair** | B: `'\ncup: chair\ncamera: queen\nanchor: horse\nbucket: teacher\ncamera:'` -> **queen**
- order 2: `'\ncamera: queen\ncup: chair\nanchor: horse\nbucket: teacher\ncup:'` -> **chair** | B: `'\ncamera: queen\ncup: chair\nanchor: horse\nbucket: teacher\ncamera:'` -> **queen**

## 15. l_ind|cup|circle|camera|castle|paper|bottle|anchor|kitchen (synthetic)
- order 1: `'\ncup: camera\ncircle: castle\npaper: bottle\nanchor: kitchen\ncup:'` -> **camera** | B: `'\ncup: camera\ncircle: castle\npaper: bottle\nanchor: kitchen\ncircle:'` -> **castle**
- order 2: `'\ncircle: castle\ncup: camera\npaper: bottle\nanchor: kitchen\ncup:'` -> **camera** | B: `'\ncircle: castle\ncup: camera\npaper: bottle\nanchor: kitchen\ncircle:'` -> **castle**

## 16. l_ind|oven|glass|ladder|cup|river|cabin|bridge|wagon (synthetic)
- order 1: `'\noven: ladder\nglass: cup\nriver: cabin\nbridge: wagon\noven:'` -> **ladder** | B: `'\noven: ladder\nglass: cup\nriver: cabin\nbridge: wagon\nglass:'` -> **cup**
- order 2: `'\nglass: cup\noven: ladder\nriver: cabin\nbridge: wagon\noven:'` -> **ladder** | B: `'\nglass: cup\noven: ladder\nriver: cabin\nbridge: wagon\nglass:'` -> **cup**

## 17. l_ind|anchor|glass|tunnel|piano|school|letter|cloud|button (synthetic)
- order 1: `'\nanchor: tunnel\nglass: piano\nschool: letter\ncloud: button\nanchor:'` -> **tunnel** | B: `'\nanchor: tunnel\nglass: piano\nschool: letter\ncloud: button\nglass:'` -> **piano**
- order 2: `'\nglass: piano\nanchor: tunnel\nschool: letter\ncloud: button\nanchor:'` -> **tunnel** | B: `'\nglass: piano\nanchor: tunnel\nschool: letter\ncloud: button\nglass:'` -> **piano**

## 18. l_ind|glass|camera|candle|button|school|doctor|oven|cup (synthetic)
- order 1: `'\nglass: candle\ncamera: button\nschool: doctor\noven: cup\nglass:'` -> **candle** | B: `'\nglass: candle\ncamera: button\nschool: doctor\noven: cup\ncamera:'` -> **button**
- order 2: `'\ncamera: button\nglass: candle\nschool: doctor\noven: cup\nglass:'` -> **candle** | B: `'\ncamera: button\nglass: candle\nschool: doctor\noven: cup\ncamera:'` -> **button**

## 19. l_ind|window|cup|cabin|island|bridge|letter|school|carpet (synthetic)
- order 1: `'\nwindow: cabin\ncup: island\nbridge: letter\nschool: carpet\nwindow:'` -> **cabin** | B: `'\nwindow: cabin\ncup: island\nbridge: letter\nschool: carpet\ncup:'` -> **island**
- order 2: `'\ncup: island\nwindow: cabin\nbridge: letter\nschool: carpet\nwindow:'` -> **cabin** | B: `'\ncup: island\nwindow: cabin\nbridge: letter\nschool: carpet\ncup:'` -> **island**

## 20. l_ind|dog|circle|sugar|blanket|cloud|desert|wallet|needle (synthetic)
- order 1: `'\ndog: sugar\ncircle: blanket\ncloud: desert\nwallet: needle\ndog:'` -> **sugar** | B: `'\ndog: sugar\ncircle: blanket\ncloud: desert\nwallet: needle\ncircle:'` -> **blanket**
- order 2: `'\ncircle: blanket\ndog: sugar\ncloud: desert\nwallet: needle\ndog:'` -> **sugar** | B: `'\ncircle: blanket\ndog: sugar\ncloud: desert\nwallet: needle\ncircle:'` -> **blanket**

## 21. l_ind|cloud|bread|guitar|farmer|anchor|valley|button|circle (synthetic)
- order 1: `'\ncloud: guitar\nbread: farmer\nanchor: valley\nbutton: circle\ncloud:'` -> **guitar** | B: `'\ncloud: guitar\nbread: farmer\nanchor: valley\nbutton: circle\nbread:'` -> **farmer**
- order 2: `'\nbread: farmer\ncloud: guitar\nanchor: valley\nbutton: circle\ncloud:'` -> **guitar** | B: `'\nbread: farmer\ncloud: guitar\nanchor: valley\nbutton: circle\nbread:'` -> **farmer**

## 22. l_ind|engine|glass|hammer|blanket|castle|forest|letter|jacket (synthetic)
- order 1: `'\nengine: hammer\nglass: blanket\ncastle: forest\nletter: jacket\nengine:'` -> **hammer** | B: `'\nengine: hammer\nglass: blanket\ncastle: forest\nletter: jacket\nglass:'` -> **blanket**
- order 2: `'\nglass: blanket\nengine: hammer\ncastle: forest\nletter: jacket\nengine:'` -> **hammer** | B: `'\nglass: blanket\nengine: hammer\ncastle: forest\nletter: jacket\nglass:'` -> **blanket**

## 23. l_ind|wallet|table|river|lemon|glass|needle|engine|anchor (synthetic)
- order 1: `'\nwallet: river\ntable: lemon\nglass: needle\nengine: anchor\nwallet:'` -> **river** | B: `'\nwallet: river\ntable: lemon\nglass: needle\nengine: anchor\ntable:'` -> **lemon**
- order 2: `'\ntable: lemon\nwallet: river\nglass: needle\nengine: anchor\nwallet:'` -> **river** | B: `'\ntable: lemon\nwallet: river\nglass: needle\nengine: anchor\ntable:'` -> **lemon**

## 24. l_ind|cloud|window|wallet|carpet|oven|village|school|pillow (synthetic)
- order 1: `'\ncloud: wallet\nwindow: carpet\noven: village\nschool: pillow\ncloud:'` -> **wallet** | B: `'\ncloud: wallet\nwindow: carpet\noven: village\nschool: pillow\nwindow:'` -> **carpet**
- order 2: `'\nwindow: carpet\ncloud: wallet\noven: village\nschool: pillow\ncloud:'` -> **wallet** | B: `'\nwindow: carpet\ncloud: wallet\noven: village\nschool: pillow\nwindow:'` -> **carpet**

## 25. l_ind|cup|paper|tunnel|valley|cloud|castle|engine|bread (synthetic)
- order 1: `'\ncup: tunnel\npaper: valley\ncloud: castle\nengine: bread\ncup:'` -> **tunnel** | B: `'\ncup: tunnel\npaper: valley\ncloud: castle\nengine: bread\npaper:'` -> **valley**
- order 2: `'\npaper: valley\ncup: tunnel\ncloud: castle\nengine: bread\ncup:'` -> **tunnel** | B: `'\npaper: valley\ncup: tunnel\ncloud: castle\nengine: bread\npaper:'` -> **valley**

## 26. l_ind|cup|oven|sugar|winter|castle|wallet|cloud|orange (synthetic)
- order 1: `'\ncup: sugar\noven: winter\ncastle: wallet\ncloud: orange\ncup:'` -> **sugar** | B: `'\ncup: sugar\noven: winter\ncastle: wallet\ncloud: orange\noven:'` -> **winter**
- order 2: `'\noven: winter\ncup: sugar\ncastle: wallet\ncloud: orange\ncup:'` -> **sugar** | B: `'\noven: winter\ncup: sugar\ncastle: wallet\ncloud: orange\noven:'` -> **winter**

## 27. l_ind|castle|school|window|garden|camera|winter|bucket|circle (synthetic)
- order 1: `'\ncastle: window\nschool: garden\ncamera: winter\nbucket: circle\ncastle:'` -> **window** | B: `'\ncastle: window\nschool: garden\ncamera: winter\nbucket: circle\nschool:'` -> **garden**
- order 2: `'\nschool: garden\ncastle: window\ncamera: winter\nbucket: circle\ncastle:'` -> **window** | B: `'\nschool: garden\ncastle: window\ncamera: winter\nbucket: circle\nschool:'` -> **garden**

## 28. l_ind|window|bridge|candle|queen|chair|lemon|letter|horse (synthetic)
- order 1: `'\nwindow: candle\nbridge: queen\nchair: lemon\nletter: horse\nwindow:'` -> **candle** | B: `'\nwindow: candle\nbridge: queen\nchair: lemon\nletter: horse\nbridge:'` -> **queen**
- order 2: `'\nbridge: queen\nwindow: candle\nchair: lemon\nletter: horse\nwindow:'` -> **candle** | B: `'\nbridge: queen\nwindow: candle\nchair: lemon\nletter: horse\nbridge:'` -> **queen**

## 29. l_ind|bucket|anchor|bridge|piano|table|queen|oven|valley (synthetic)
- order 1: `'\nbucket: bridge\nanchor: piano\ntable: queen\noven: valley\nbucket:'` -> **bridge** | B: `'\nbucket: bridge\nanchor: piano\ntable: queen\noven: valley\nanchor:'` -> **piano**
- order 2: `'\nanchor: piano\nbucket: bridge\ntable: queen\noven: valley\nbucket:'` -> **bridge** | B: `'\nanchor: piano\nbucket: bridge\ntable: queen\noven: valley\nanchor:'` -> **piano**

## 30. l_ind|chair|bucket|blanket|hammer|school|castle|dog|bottle (synthetic)
- order 1: `'\nchair: blanket\nbucket: hammer\nschool: castle\ndog: bottle\nchair:'` -> **blanket** | B: `'\nchair: blanket\nbucket: hammer\nschool: castle\ndog: bottle\nbucket:'` -> **hammer**
- order 2: `'\nbucket: hammer\nchair: blanket\nschool: castle\ndog: bottle\nchair:'` -> **blanket** | B: `'\nbucket: hammer\nchair: blanket\nschool: castle\ndog: bottle\nbucket:'` -> **hammer**

## 31. l_ind|glass|cloud|bottle|castle|circle|sugar|dog|chair (synthetic)
- order 1: `'\nglass: bottle\ncloud: castle\ncircle: sugar\ndog: chair\nglass:'` -> **bottle** | B: `'\nglass: bottle\ncloud: castle\ncircle: sugar\ndog: chair\ncloud:'` -> **castle**
- order 2: `'\ncloud: castle\nglass: bottle\ncircle: sugar\ndog: chair\nglass:'` -> **bottle** | B: `'\ncloud: castle\nglass: bottle\ncircle: sugar\ndog: chair\ncloud:'` -> **castle**

## 32. l_ind|cup|bucket|guitar|pillow|letter|table|anchor|yellow (synthetic)
- order 1: `'\ncup: guitar\nbucket: pillow\nletter: table\nanchor: yellow\ncup:'` -> **guitar** | B: `'\ncup: guitar\nbucket: pillow\nletter: table\nanchor: yellow\nbucket:'` -> **pillow**
- order 2: `'\nbucket: pillow\ncup: guitar\nletter: table\nanchor: yellow\ncup:'` -> **guitar** | B: `'\nbucket: pillow\ncup: guitar\nletter: table\nanchor: yellow\nbucket:'` -> **pillow**

## 33. l_ind|bucket|bread|saddle|table|circle|kitchen|wallet|basket (synthetic)
- order 1: `'\nbucket: saddle\nbread: table\ncircle: kitchen\nwallet: basket\nbucket:'` -> **saddle** | B: `'\nbucket: saddle\nbread: table\ncircle: kitchen\nwallet: basket\nbread:'` -> **table**
- order 2: `'\nbread: table\nbucket: saddle\ncircle: kitchen\nwallet: basket\nbucket:'` -> **saddle** | B: `'\nbread: table\nbucket: saddle\ncircle: kitchen\nwallet: basket\nbread:'` -> **table**

## 34. l_ind|window|glass|cup|tunnel|table|castle|button|jacket (synthetic)
- order 1: `'\nwindow: cup\nglass: tunnel\ntable: castle\nbutton: jacket\nwindow:'` -> **cup** | B: `'\nwindow: cup\nglass: tunnel\ntable: castle\nbutton: jacket\nglass:'` -> **tunnel**
- order 2: `'\nglass: tunnel\nwindow: cup\ntable: castle\nbutton: jacket\nwindow:'` -> **cup** | B: `'\nglass: tunnel\nwindow: cup\ntable: castle\nbutton: jacket\nglass:'` -> **tunnel**

## 35. l_ind|button|window|cabin|mirror|bucket|forest|bread|ocean (synthetic)
- order 1: `'\nbutton: cabin\nwindow: mirror\nbucket: forest\nbread: ocean\nbutton:'` -> **cabin** | B: `'\nbutton: cabin\nwindow: mirror\nbucket: forest\nbread: ocean\nwindow:'` -> **mirror**
- order 2: `'\nwindow: mirror\nbutton: cabin\nbucket: forest\nbread: ocean\nbutton:'` -> **cabin** | B: `'\nwindow: mirror\nbutton: cabin\nbucket: forest\nbread: ocean\nwindow:'` -> **mirror**

## 36. l_ind|bridge|oven|carpet|basket|river|ladder|window|orange (synthetic)
- order 1: `'\nbridge: carpet\noven: basket\nriver: ladder\nwindow: orange\nbridge:'` -> **carpet** | B: `'\nbridge: carpet\noven: basket\nriver: ladder\nwindow: orange\noven:'` -> **basket**
- order 2: `'\noven: basket\nbridge: carpet\nriver: ladder\nwindow: orange\nbridge:'` -> **carpet** | B: `'\noven: basket\nbridge: carpet\nriver: ladder\nwindow: orange\noven:'` -> **basket**

## 37. l_ind|table|letter|chair|ocean|glass|oven|camera|garden (synthetic)
- order 1: `'\ntable: chair\nletter: ocean\nglass: oven\ncamera: garden\ntable:'` -> **chair** | B: `'\ntable: chair\nletter: ocean\nglass: oven\ncamera: garden\nletter:'` -> **ocean**
- order 2: `'\nletter: ocean\ntable: chair\nglass: oven\ncamera: garden\ntable:'` -> **chair** | B: `'\nletter: ocean\ntable: chair\nglass: oven\ncamera: garden\nletter:'` -> **ocean**

## 38. l_ind|oven|school|garden|bread|window|forest|anchor|castle (synthetic)
- order 1: `'\noven: garden\nschool: bread\nwindow: forest\nanchor: castle\noven:'` -> **garden** | B: `'\noven: garden\nschool: bread\nwindow: forest\nanchor: castle\nschool:'` -> **bread**
- order 2: `'\nschool: bread\noven: garden\nwindow: forest\nanchor: castle\noven:'` -> **garden** | B: `'\nschool: bread\noven: garden\nwindow: forest\nanchor: castle\nschool:'` -> **bread**

## 39. l_ind|table|button|mountain|jacket|wallet|needle|window|desert (synthetic)
- order 1: `'\ntable: mountain\nbutton: jacket\nwallet: needle\nwindow: desert\ntable:'` -> **mountain** | B: `'\ntable: mountain\nbutton: jacket\nwallet: needle\nwindow: desert\nbutton:'` -> **jacket**
- order 2: `'\nbutton: jacket\ntable: mountain\nwallet: needle\nwindow: desert\ntable:'` -> **mountain** | B: `'\nbutton: jacket\ntable: mountain\nwallet: needle\nwindow: desert\nbutton:'` -> **jacket**

## 40. l_ind|engine|circle|teacher|mountain|school|ladder|camera|hammer (synthetic)
- order 1: `'\nengine: teacher\ncircle: mountain\nschool: ladder\ncamera: hammer\nengine:'` -> **teacher** | B: `'\nengine: teacher\ncircle: mountain\nschool: ladder\ncamera: hammer\ncircle:'` -> **mountain**
- order 2: `'\ncircle: mountain\nengine: teacher\nschool: ladder\ncamera: hammer\nengine:'` -> **teacher** | B: `'\ncircle: mountain\nengine: teacher\nschool: ladder\ncamera: hammer\ncircle:'` -> **mountain**

## 41. l_ind|cup|bucket|farmer|saddle|letter|dog|table|mirror (synthetic)
- order 1: `'\ncup: farmer\nbucket: saddle\nletter: dog\ntable: mirror\ncup:'` -> **farmer** | B: `'\ncup: farmer\nbucket: saddle\nletter: dog\ntable: mirror\nbucket:'` -> **saddle**
- order 2: `'\nbucket: saddle\ncup: farmer\nletter: dog\ntable: mirror\ncup:'` -> **farmer** | B: `'\nbucket: saddle\ncup: farmer\nletter: dog\ntable: mirror\nbucket:'` -> **saddle**

## 42. l_ind|dog|cloud|horse|letter|cup|queen|anchor|river (synthetic)
- order 1: `'\ndog: horse\ncloud: letter\ncup: queen\nanchor: river\ndog:'` -> **horse** | B: `'\ndog: horse\ncloud: letter\ncup: queen\nanchor: river\ncloud:'` -> **letter**
- order 2: `'\ncloud: letter\ndog: horse\ncup: queen\nanchor: river\ndog:'` -> **horse** | B: `'\ncloud: letter\ndog: horse\ncup: queen\nanchor: river\ncloud:'` -> **letter**

## 43. l_ind|bridge|paper|candle|basket|dog|saddle|river|teacher (synthetic)
- order 1: `'\nbridge: candle\npaper: basket\ndog: saddle\nriver: teacher\nbridge:'` -> **candle** | B: `'\nbridge: candle\npaper: basket\ndog: saddle\nriver: teacher\npaper:'` -> **basket**
- order 2: `'\npaper: basket\nbridge: candle\ndog: saddle\nriver: teacher\nbridge:'` -> **candle** | B: `'\npaper: basket\nbridge: candle\ndog: saddle\nriver: teacher\npaper:'` -> **basket**

## 44. l_ind|button|glass|hammer|bridge|oven|planet|castle|bucket (synthetic)
- order 1: `'\nbutton: hammer\nglass: bridge\noven: planet\ncastle: bucket\nbutton:'` -> **hammer** | B: `'\nbutton: hammer\nglass: bridge\noven: planet\ncastle: bucket\nglass:'` -> **bridge**
- order 2: `'\nglass: bridge\nbutton: hammer\noven: planet\ncastle: bucket\nbutton:'` -> **hammer** | B: `'\nglass: bridge\nbutton: hammer\noven: planet\ncastle: bucket\nglass:'` -> **bridge**

## 45. l_ind|wallet|dog|planet|button|bucket|table|oven|hammer (synthetic)
- order 1: `'\nwallet: planet\ndog: button\nbucket: table\noven: hammer\nwallet:'` -> **planet** | B: `'\nwallet: planet\ndog: button\nbucket: table\noven: hammer\ndog:'` -> **button**
- order 2: `'\ndog: button\nwallet: planet\nbucket: table\noven: hammer\nwallet:'` -> **planet** | B: `'\ndog: button\nwallet: planet\nbucket: table\noven: hammer\ndog:'` -> **button**

## 46. l_ind|bread|cloud|queen|bucket|letter|circle|engine|glass (synthetic)
- order 1: `'\nbread: queen\ncloud: bucket\nletter: circle\nengine: glass\nbread:'` -> **queen** | B: `'\nbread: queen\ncloud: bucket\nletter: circle\nengine: glass\ncloud:'` -> **bucket**
- order 2: `'\ncloud: bucket\nbread: queen\nletter: circle\nengine: glass\nbread:'` -> **queen** | B: `'\ncloud: bucket\nbread: queen\nletter: circle\nengine: glass\ncloud:'` -> **bucket**

## 47. l_ind|cup|circle|saddle|chair|button|letter|dog|pillow (synthetic)
- order 1: `'\ncup: saddle\ncircle: chair\nbutton: letter\ndog: pillow\ncup:'` -> **saddle** | B: `'\ncup: saddle\ncircle: chair\nbutton: letter\ndog: pillow\ncircle:'` -> **chair**
- order 2: `'\ncircle: chair\ncup: saddle\nbutton: letter\ndog: pillow\ncup:'` -> **saddle** | B: `'\ncircle: chair\ncup: saddle\nbutton: letter\ndog: pillow\ncircle:'` -> **chair**

## 48. l_ind|school|bridge|oven|bucket|window|desert|wallet|queen (synthetic)
- order 1: `'\nschool: oven\nbridge: bucket\nwindow: desert\nwallet: queen\nschool:'` -> **oven** | B: `'\nschool: oven\nbridge: bucket\nwindow: desert\nwallet: queen\nbridge:'` -> **bucket**
- order 2: `'\nbridge: bucket\nschool: oven\nwindow: desert\nwallet: queen\nschool:'` -> **oven** | B: `'\nbridge: bucket\nschool: oven\nwindow: desert\nwallet: queen\nbridge:'` -> **bucket**

## 49. l_ind|button|camera|window|forest|letter|cabin|river|mirror (synthetic)
- order 1: `'\nbutton: window\ncamera: forest\nletter: cabin\nriver: mirror\nbutton:'` -> **window** | B: `'\nbutton: window\ncamera: forest\nletter: cabin\nriver: mirror\ncamera:'` -> **forest**
- order 2: `'\ncamera: forest\nbutton: window\nletter: cabin\nriver: mirror\nbutton:'` -> **window** | B: `'\ncamera: forest\nbutton: window\nletter: cabin\nriver: mirror\ncamera:'` -> **forest**

## 50. l_ind|letter|button|forest|bucket|circle|ocean|castle|bottle (synthetic)
- order 1: `'\nletter: forest\nbutton: bucket\ncircle: ocean\ncastle: bottle\nletter:'` -> **forest** | B: `'\nletter: forest\nbutton: bucket\ncircle: ocean\ncastle: bottle\nbutton:'` -> **bucket**
- order 2: `'\nbutton: bucket\nletter: forest\ncircle: ocean\ncastle: bottle\nletter:'` -> **forest** | B: `'\nbutton: bucket\nletter: forest\ncircle: ocean\ncastle: bottle\nbutton:'` -> **bucket**
