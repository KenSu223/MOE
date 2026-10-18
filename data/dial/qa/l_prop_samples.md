# l_prop: 50 seeded samples (qwen-family texts; random_state 0)

## 1. prop|object_taste|bacon|coffee (curated, object_taste)
- template 1: `'The bacon tastes'` -> **salty** | B: `'The coffee tastes'` -> **bitter**

## 2. prop|object_color|coal|carrot (curated, object_color)
- template 1: `'Q: What color is the coal?\nA:'` -> **black** | B: `'Q: What color is the carrot?\nA:'` -> **orange**

## 3. prop|object_taste|salt|pepper (curated, object_taste)
- template 1: `'The salt tastes'` -> **salty** | B: `'The pepper tastes'` -> **spicy**

## 4. prop|object_color|peach|sky (curated, object_color)
- template 1: `'Q: What color is the peach?\nA:'` -> **pink** | B: `'Q: What color is the sky?\nA:'` -> **blue**

## 5. prop|object_material|shirt|scarf (curated, object_material)
- template 0: `'The shirt is made of'` -> **cotton** | B: `'The scarf is made of'` -> **wool**
- template 1: `'A typical shirt is made of'` -> **cotton** | B: `'A typical scarf is made of'` -> **wool**

## 6. prop|object_color|tar|plum (curated, object_color)
- template 1: `'Q: What color is the tar?\nA:'` -> **black** | B: `'Q: What color is the plum?\nA:'` -> **purple**

## 7. prop|work_location|researcher|lawyer (lre, work_location)
- template 1: `'You can usually find a researcher working in a'` -> **laboratory** | B: `'You can usually find a lawyer working in a'` -> **courthouse**

## 8. prop|substance_phase|wine|wax (lre, substance_phase)
- template 0: `'At room temperature, wine is a'` -> **liquid** | B: `'At room temperature, wax is a'` -> **solid**

## 9. prop|substance_phase|coffee|wood (lre, substance_phase)
- template 0: `'At room temperature, coffee is a'` -> **liquid** | B: `'At room temperature, wood is a'` -> **solid**

## 10. prop|substance_phase|leather|neon (lre, substance_phase)
- template 0: `'At room temperature, leather is a'` -> **solid** | B: `'At room temperature, neon is a'` -> **gas**

## 11. prop|object_color|milk|charcoal (curated, object_color)
- template 0: `'The color of the milk is'` -> **white** | B: `'The color of the charcoal is'` -> **black**

## 12. prop|object_material|door|sweater (curated, object_material)
- template 0: `'The door is made of'` -> **wood** | B: `'The sweater is made of'` -> **wool**

## 13. prop|substance_phase|tea|plastic (lre, substance_phase)
- template 1: `'What is the phase of matter of tea at room temperature? It is a'` -> **liquid** | B: `'What is the phase of matter of plastic at room temperature? It is a'` -> **solid**

## 14. prop|work_location|farmer|photographer (lre, work_location)
- template 1: `'You can usually find a farmer working in a'` -> **farm** | B: `'You can usually find a photographer working in a'` -> **studio**

## 15. prop|work_location|journalist|chef (lre, work_location)
- template 1: `'You can usually find a journalist working in a'` -> **office** | B: `'You can usually find a chef working in a'` -> **kitchen**

## 16. prop|object_color|chocolate|ocean (curated, object_color)
- template 0: `'The color of the chocolate is'` -> **brown** | B: `'The color of the ocean is'` -> **blue**

## 17. prop|substance_phase|paper|nitrogen (lre, substance_phase)
- template 1: `'What is the phase of matter of paper at room temperature? It is a'` -> **solid** | B: `'What is the phase of matter of nitrogen at room temperature? It is a'` -> **gas**

## 18. prop|object_material|sweater|spoon (curated, object_material)
- template 1: `'A typical sweater is made of'` -> **wool** | B: `'A typical spoon is made of'` -> **metal**

## 19. prop|object_superclass|turkey|pear (lre, object_superclass)
- template 1: `'A turkey is a kind of'` -> **bird** | B: `'A pear is a kind of'` -> **fruit**

## 20. prop|object_taste|jam|olive (curated, object_taste)
- template 1: `'The jam tastes'` -> **sweet** | B: `'The olive tastes'` -> **salty**

## 21. prop|substance_phase|nitrogen|tea (lre, substance_phase)
- template 1: `'What is the phase of matter of nitrogen at room temperature? It is a'` -> **gas** | B: `'What is the phase of matter of tea at room temperature? It is a'` -> **liquid**

## 22. prop|object_color|charcoal|butter (curated, object_color)
- template 0: `'The color of the charcoal is'` -> **black** | B: `'The color of the butter is'` -> **yellow**

## 23. prop|object_superclass|carrot|flower (lre, object_superclass)
- template 0: `'A carrot is in the category of a'` -> **vegetable** | B: `'A flower is in the category of a'` -> **plant**

## 24. prop|object_color|taxi|peach (curated, object_color)
- template 1: `'Q: What color is the taxi?\nA:'` -> **yellow** | B: `'Q: What color is the peach?\nA:'` -> **pink**

## 25. prop|object_superclass|peach|broccoli (lre, object_superclass)
- template 1: `'A peach is a kind of'` -> **fruit** | B: `'A broccoli is a kind of'` -> **vegetable**

## 26. prop|object_color|cucumber|butter (curated, object_color)
- template 1: `'Q: What color is the cucumber?\nA:'` -> **green** | B: `'Q: What color is the butter?\nA:'` -> **yellow**

## 27. prop|work_location|nurse|baker (lre, work_location)
- template 0: `'A nurse typically works at a'` -> **hospital** | B: `'A baker typically works at a'` -> **bakery**

## 28. prop|animal_habitat|sheep|whale (curated, animal_habitat)
- template 1: `'The natural home of the sheep is the'` -> **farm** | B: `'The natural home of the whale is the'` -> **ocean**

## 29. prop|fruit_inside_color|blackberries|blueberries (lre, fruit_inside_color)
- template 0: `'On the inside, blackberries are'` -> **red** | B: `'On the inside, blueberries are'` -> **green**
- template 1: `'What color are blackberries on the inside? They are'` -> **red** | B: `'What color are blueberries on the inside? They are'` -> **green**

## 30. prop|object_color|rose|grape (curated, object_color)
- template 0: `'The color of the rose is'` -> **red** | B: `'The color of the grape is'` -> **purple**

## 31. prop|object_superclass|bass|potato (lre, object_superclass)
- template 0: `'A bass is in the category of a'` -> **fish** | B: `'A potato is in the category of a'` -> **vegetable**

## 32. prop|object_color|avocado|plum (curated, object_color)
- template 0: `'The color of the avocado is'` -> **green** | B: `'The color of the plum is'` -> **purple**

## 33. prop|object_material|brick|newspaper (curated, object_material)
- template 1: `'A typical brick is made of'` -> **clay** | B: `'A typical newspaper is made of'` -> **paper**

## 34. prop|object_superclass|duck|flower (lre, object_superclass)
- template 0: `'A duck is in the category of a'` -> **bird** | B: `'A flower is in the category of a'` -> **plant**
- template 1: `'A duck is a kind of'` -> **bird** | B: `'A flower is a kind of'` -> **plant**

## 35. prop|animal_habitat|tiger|shark (curated, animal_habitat)
- template 0: `'The tiger lives in the'` -> **jungle** | B: `'The shark lives in the'` -> **ocean**

## 36. prop|object_taste|cake|curry (curated, object_taste)
- template 0: `'The taste of the cake is'` -> **sweet** | B: `'The taste of the curry is'` -> **spicy**
- template 1: `'The cake tastes'` -> **sweet** | B: `'The curry tastes'` -> **spicy**

## 37. prop|object_material|tire|pencil (curated, object_material)
- template 0: `'The tire is made of'` -> **rubber** | B: `'The pencil is made of'` -> **wood**

## 38. prop|object_color|crow|carrot (curated, object_color)
- template 0: `'The color of the crow is'` -> **black** | B: `'The color of the carrot is'` -> **orange**

## 39. prop|work_location|doctor|student (lre, work_location)
- template 1: `'You can usually find a doctor working in a'` -> **hospital** | B: `'You can usually find a student working in a'` -> **school**

## 40. prop|fruit_outside_color|raspberries|blueberries (lre, fruit_outside_color)
- template 0: `'On the outside, raspberries are'` -> **red** | B: `'On the outside, blueberries are'` -> **blue**
- template 1: `'The color of the skin of raspberries is'` -> **red** | B: `'The color of the skin of blueberries is'` -> **blue**

## 41. prop|substance_phase|juice|wax (lre, substance_phase)
- template 0: `'At room temperature, juice is a'` -> **liquid** | B: `'At room temperature, wax is a'` -> **solid**

## 42. prop|object_color|cherry|pumpkin (curated, object_color)
- template 0: `'The color of the cherry is'` -> **red** | B: `'The color of the pumpkin is'` -> **orange**

## 43. prop|object_color|sky|pig (curated, object_color)
- template 0: `'The color of the sky is'` -> **blue** | B: `'The color of the pig is'` -> **pink**

## 44. prop|animal_habitat|monkey|cow (curated, animal_habitat)
- template 1: `'The natural home of the monkey is the'` -> **jungle** | B: `'The natural home of the cow is the'` -> **farm**

## 45. prop|object_superclass|grapes|broccoli (lre, object_superclass)
- template 0: `'A grapes is in the category of a'` -> **fruit** | B: `'A broccoli is in the category of a'` -> **vegetable**

## 46. prop|object_superclass|grapes|potato (lre, object_superclass)
- template 1: `'A grapes is a kind of'` -> **fruit** | B: `'A potato is a kind of'` -> **vegetable**

## 47. prop|object_color|squirrel|pearl (curated, object_color)
- template 0: `'The color of the squirrel is'` -> **brown** | B: `'The color of the pearl is'` -> **white**

## 48. prop|work_location|athlete|researcher (lre, work_location)
- template 1: `'You can usually find a athlete working in a'` -> **stadium** | B: `'You can usually find a researcher working in a'` -> **laboratory**

## 49. prop|work_location|artist|teacher (lre, work_location)
- template 1: `'You can usually find a artist working in a'` -> **studio** | B: `'You can usually find a teacher working in a'` -> **school**

## 50. prop|animal_habitat|monkey|trout (curated, animal_habitat)
- template 1: `'The natural home of the monkey is the'` -> **jungle** | B: `'The natural home of the trout is the'` -> **river**
