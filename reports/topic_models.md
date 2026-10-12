# Topic Model Comparison and Tuning Sweep

Fit on the 15824 reviews of the training split. All configurations
scored on c_v coherence over the same tokenized corpus and dictionary.

## Sweep results

| model    | config                            | n_topics | coherence_cv | outlier_docs | outlier_pct |
|----------|-----------------------------------|----------|--------------|--------------|-------------|
| LDA      | num_topics=10                     | 10       | 0.4701       |              |             |
| BERTopic | min_topic_size=25, nr_topics=None | 53       | 0.4543       | 5108         | 32.3        |
| BERTopic | min_topic_size=50, nr_topics=10   | 9        | 0.4039       | 1284         | 8.1         |

Best BERTopic: **min_topic_size=25, nr_topics=None** (c_v 0.4543)
Best LDA: **num_topics=10** (c_v 0.4701)

## BERTopic topics — min_topic_size=25, nr_topics=None

| Topic | Count | Top words                                                                                                                                  |
|-------|-------|--------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | 3225  | dress, size, love dress, fit, love, fabric, wear, like, beautiful, just                                                                    |
| 1     | 845   | shirt, love shirt, shirts, love, great, soft, cute, wear, little, bought                                                                   |
| 2     | 701   | sweater, love sweater, soft, color, love, sweaters, sleeves, like, warm, just                                                              |
| 3     | 565   | skirt, waist, skirts, size, great, hips, length, love, fabric, really                                                                      |
| 4     | 536   | love, great, cute, looks, soft, comfortable, jeans, like, fabric, colors                                                                   |
| 5     | 480   | xs, petite, xxs, dress, xs petite, regular, lbs, ordered, small, fit                                                                       |
| 6     | 421   | blouse, blouses, size, wear, love, fabric, just, beautiful, white, small                                                                   |
| 7     | 237   | tee, tees, bought, white, love, soft, color, nice, colors, great                                                                           |
| 8     | 217   | tank, great, straps, wear, love, tanks, wear tank, white, colors, tank underneath                                                          |
| 9     | 205   | great, piece, fabric, quality, love, runs, material, fit, fits, nice                                                                       |
| 10    | 188   | pants, love pants, comfortable, legs, fit, pair, great, love, like, pants comfortable                                                      |
| 11    | 160   | jeans, love jeans, stretch, rise, jean, fit, pair, love, great, perfect                                                                    |
| 12    | 156   | jacket, love jacket, love, denim, great, jackets, perfect, wear, sleeves, jacket cute                                                      |
| 13    | 148   | tunic, love tunic, leggings, jeans, wear, skinny jeans, like, jeans leggings, skinny, love                                                 |
| 14    | 147   | cardigan, cardigans, sleeves, soft, like, love, sweater, long, great, perfect                                                              |
| 15    | 146   | medium, tops, short, small, size, love, lbs, ordered, torso, fit                                                                           |
| 16    | 136   | xs, xxs, small, tops, love, lbs, cute, ordered xs, xs fit, size                                                                            |
| 17    | 133   | suit, swimsuit, bathing, bathing suit, bottoms, coverage, support, bikini, suits, size                                                     |
| 18    | 132   | vest, wear, love, soft, warm, color, just, vests, collar, fur                                                                              |
| 19    | 127   | jumpsuit, jumper, torso, love, fit, small, waist, time, wear, easy                                                                         |
| 20    | 125   | 28, 27, 29, jeans, 26, 25, fit, pants, size, stretch                                                                                       |
| 21    | 113   | pilcro, jeans, pilcro jeans, pair, pants, fit, pairs, pilcro pants, stretch, love                                                          |
| 22    | 110   | bust, chest, chested, didn, tops, look, large, work, small, smaller                                                                        |
| 23    | 86    | pink, orange, red, color, mustard, green, got, online, pink color, looks                                                                   |
| 24    | 86    | romper, rompers, cute, shorts, long, short, love, torso, small, flattering                                                                 |
| 25    | 84    | shorts, short, perfect, fit, length, legs, great, hard, comfortable, love                                                                  |
| 26    | 75    | soft, super soft, super, soft comfortable, fabric, soft comfy, material, comfy, material soft, great                                       |
| 27    | 72    | coat, warm, trench, unlined, rain, amp, medium, like, winter, love                                                                         |
| 28    | 71    | leggings, legging, warm, comfortable, black, price, tights, pair, feel, great                                                              |
| 29    | 70    | sweater, xs, ordered xs, arms, xxs, sleeves, long, ordered, petite, length                                                                 |
| 30    | 70    | itchy, sweater, wool, itch, cute, amp, quot, really, like, long                                                                            |
| 31    | 69    | xl, size xl, ordered, matches, true size, size, true, photos, xl fit, version                                                              |
| 32    | 52    | pants, love pants, length, long, look, waist, petite, ankles, love, way                                                                    |
| 33    | 52    | cami, need cami, cami underneath, need, wear cami, underneath, sheer, mesh, think, wear                                                    |
| 34    | 50    | comfy, pair, great, 70s, super comfy, love, winter, flattering, look great, comfortable                                                    |
| 35    | 45    | blazer, jacket, look, size, wear, small, casual, wardrobe, buttons, boxy                                                                   |
| 36    | 44    | white, bought white, sheer, ordered white, bought, shirt white, shirt, dryer, ordered, worn                                                |
| 37    | 41    | jacket, xs, small, jackets, size, ordered, like, snug, wear, little                                                                        |
| 38    | 39    | runs, runs large, large, does run, big, run, runs little, say runs, reviews, yes                                                           |
| 39    | 39    | hanger, dress, tried, look, looked, looks, body, store, dresses, really                                                                    |
| 40    | 38    | medium, large, ordered, medium fits, medium fit, size, fits, med, ordered large, sleeve shirt                                              |
| 41    | 37    | black, got black, bought black, bought, white, great, fits, got, blue, ordered black                                                       |
| 42    | 37    | ag, jeans, ag jeans, stevie, ag stevie, pair, stretch, pairs, jeans fit, rise                                                              |
| 43    | 36    | bra, bras, cups, straps, padding, racerback, like, support, bra underneath, cup                                                            |
| 44    | 35    | comfortable, comfy, comfortable looks, comfortable great, cute comfortable, versatile, great, comfy great, ve owned, comfortable versatile |
| 45    | 33    | jacket, medium, jackets, small, lbs, fits, tight, large, size, sleeves                                                                     |
| 46    | 31    | poncho, center, fall, oversized, love, underneath, looks, purchase, itchy, tank                                                            |
| 47    | 30    | dry, washed, shrunk, wash, cold, cold water, washing, water, washed cold, instructions                                                     |
| 48    | 30    | tts, fit tts, fits tts, got, fit, model, navy, threads, ordered, sizing                                                                    |
| 49    | 30    | cords, grey, fit, pair, rise, color, ag, navy, 26, nice                                                                                    |
| 50    | 28    | buttons, button, dress, missing, reinforce, lost, holes, arrived, tried home, unbuttoning                                                  |
| 51    | 27    | true size, size, true, fits true, fits, large, size fits, perfect, recommend, love                                                         |
| 52    | 26    | swing, dress, swing style, dresses, style, fabric, voluminous, version, moo, pxxs                                                          |

## LDA topics — num_topics=10

 0. dress, look, make, fabric, fit, beautiful, like, flatter, would, body
 1. skirt, look, fabric, waist, back, side, like, front, pocket, nice
 2. price, sale, get, review, would, try, go, quality, store, one
 3. look, like, back, shirt, top, really, fabric, would, much, wash
 4. long, short, waist, length, hit, top, cut, sleeve, low, little
 5. dress, wear, perfect, summer, great, love, comfortable, fall, work, day
 6. size, fit, small, im, order, top, large, run, wear, petite
 7. wear, jean, love, pant, great, get, look, buy, pair, compliment
 8. love, sweater, soft, great, top, shirt, color, fit, detail, nice
 9. color, blue, buy, fit, love, green, red, black, soft, nice

## Topics by department

Top 5 topics per department, outlier documents excluded.

| department | topic | reviews | share_of_dept | top_words                                      |
|------------|-------|---------|---------------|------------------------------------------------|
| Bottoms    | 3     | 517     | 30.9          | skirt, waist, skirts, size, great              |
| Bottoms    | 10    | 160     | 9.6           | pants, love pants, comfortable, legs, fit      |
| Bottoms    | 11    | 158     | 9.4           | jeans, love jeans, stretch, rise, jean         |
| Bottoms    | 20    | 122     | 7.3           | 28, 27, 29, jeans, 26                          |
| Bottoms    | 21    | 113     | 6.7           | pilcro, jeans, pilcro jeans, pair, pants       |
| Dresses    | 0     | 2806    | 84.1          | dress, size, love dress, fit, love             |
| Dresses    | 5     | 271     | 8.1           | xs, petite, xxs, dress, xs petite              |
| Dresses    | 9     | 42      | 1.3           | great, piece, fabric, quality, love            |
| Dresses    | 39    | 25      | 0.7           | hanger, dress, tried, look, looked             |
| Dresses    | 52    | 23      | 0.7           | swing, dress, swing style, dresses, style      |
| Intimate   | 17    | 126     | 19.0          | suit, swimsuit, bathing, bathing suit, bottoms |
| Intimate   | 0     | 121     | 18.3          | dress, size, love dress, fit, love             |
| Intimate   | 28    | 54      | 8.1           | leggings, legging, warm, comfortable, black    |
| Intimate   | 8     | 49      | 7.4           | tank, great, straps, wear, love                |
| Intimate   | 5     | 28      | 4.2           | xs, petite, xxs, dress, xs petite              |
| Jackets    | 12    | 120     | 27.1          | jacket, love jacket, love, denim, great        |
| Jackets    | 18    | 72      | 16.3          | vest, wear, love, soft, warm                   |
| Jackets    | 27    | 61      | 13.8          | coat, warm, trench, unlined, rain              |
| Jackets    | 35    | 37      | 8.4           | blazer, jacket, look, size, wear               |
| Jackets    | 45    | 28      | 6.3           | jacket, medium, jackets, small, lbs            |
| Tops       | 1     | 821     | 18.1          | shirt, love shirt, shirts, love, great         |
| Tops       | 2     | 674     | 14.9          | sweater, love sweater, soft, color, love       |
| Tops       | 4     | 513     | 11.3          | love, great, cute, looks, soft                 |
| Tops       | 6     | 412     | 9.1           | blouse, blouses, size, wear, love              |
| Tops       | 7     | 225     | 5.0           | tee, tees, bought, white, love                 |
| Trend      | 0     | 30      | 49.2          | dress, size, love dress, fit, love             |
| Trend      | 3     | 9       | 14.8          | skirt, waist, skirts, size, great              |
| Trend      | 19    | 3       | 4.9           | jumpsuit, jumper, torso, love, fit             |
| Trend      | 2     | 2       | 3.3           | sweater, love sweater, soft, color, love       |
| Trend      | 5     | 2       | 3.3           | xs, petite, xxs, dress, xs petite              |
