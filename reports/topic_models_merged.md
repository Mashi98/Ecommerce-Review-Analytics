# BERTopic near-duplicate merge

Primary fit (min_topic_size 25, auto topic count) on the 15824 training reviews.
Topics merge when every pair in the group has c-TF-IDF cosine >= 0.45
(complete linkage); topics of more than 500 reviews are never merged.

## Effect on the model

| fit          | n_topics | coherence_cv | outlier_pct |
|--------------|----------|--------------|-------------|
| before merge | 53       | 0.4543       | 32.3        |
| after merge  | 45       | 0.4536       | 32.3        |

## What was merged

| group | original_topic | size | top_words                            |
|-------|----------------|------|--------------------------------------|
| 1     | 5              | 480  | xs, petite, xxs, dress               |
| 1     | 16             | 136  | xs, xxs, small, tops                 |
| 1     | 29             | 70   | sweater, xs, ordered xs, arms        |
| 2     | 10             | 188  | pants, love pants, comfortable, legs |
| 2     | 32             | 52   | pants, love pants, length, long      |
| 3     | 11             | 160  | jeans, love jeans, stretch, rise     |
| 3     | 20             | 125  | 28, 27, 29, jeans                    |
| 3     | 21             | 113  | pilcro, jeans, pilcro jeans, pair    |
| 4     | 12             | 156  | jacket, love jacket, love, denim     |
| 4     | 37             | 41   | jacket, xs, small, jackets           |
| 4     | 45             | 33   | jacket, medium, jackets, small       |
| 5     | 15             | 146  | medium, tops, short, small           |
| 5     | 22             | 110  | bust, chest, chested, didn           |

## Topics after merge

| Topic | Count | theme                                          | Top words                                                                                                                                  |
|-------|-------|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| 0     | 3225  | dresses (general)                              | dress, size, love dress, love, fit, fabric, wear, like, beautiful, just                                                                    |
| 1     | 845   | shirts                                         | shirt, love shirt, shirts, love, great, soft, cute, wear, little, bought                                                                   |
| 2     | 701   | sweaters                                       | sweater, love sweater, soft, color, love, sweaters, sleeves, like, just, warm                                                              |
| 3     | 686   | xs / petite sizing (dresses, tops, sweaters)   | xs, petite, xxs, dress, ordered, lbs, small, regular, fit, xs petite                                                                       |
| 4     | 565   | skirts and shorts (waist fit)                  | skirt, waist, skirts, size, great, love, hips, length, fabric, really                                                                      |
| 5     | 536   | generic praise for tops                        | love, great, cute, looks, soft, comfortable, like, jeans, fabric, color                                                                    |
| 6     | 421   | blouses                                        | blouse, size, blouses, wear, love, fabric, just, beautiful, white, like                                                                    |
| 7     | 398   | jeans (sizing and stretch)                     | jeans, pilcro, fit, stretch, 28, 27, pair, 29, pants, size                                                                                 |
| 8     | 256   | tops: bust and chest fit                       | bust, tops, small, medium, chest, large, size, look, fit, really                                                                           |
| 9     | 240   | pants                                          | pants, comfortable, look, love, fit, like, legs, great, waist, pants great                                                                 |
| 10    | 237   | tees                                           | tee, tees, bought, white, love, soft, color, nice, colors, great                                                                           |
| 11    | 230   | jackets                                        | jacket, jackets, love, small, denim, xs, sleeves, like, size, perfect                                                                      |
| 12    | 217   | tanks                                          | tank, great, love, wear, straps, tanks, wear tank, white, colors, underneath                                                               |
| 13    | 205   | quality, price and everyday basics             | great, piece, fabric, quality, love, fit, runs, material, fits, nice                                                                       |
| 14    | 148   | tunics and leggings                            | tunic, love tunic, leggings, jeans, wear, like, skinny jeans, love, jeans leggings, skinny                                                 |
| 15    | 147   | cardigans                                      | cardigan, cardigans, sleeves, soft, like, love, sweater, long, great, perfect                                                              |
| 16    | 133   | swimsuits                                      | suit, swimsuit, bathing, bathing suit, bottoms, coverage, support, bikini, suits, size                                                     |
| 17    | 132   | vests                                          | vest, wear, love, soft, warm, color, just, xs, collar, vests                                                                               |
| 18    | 127   | jumpsuits                                      | jumpsuit, jumper, love, torso, fit, small, wear, waist, fabric, time                                                                       |
| 19    | 86    | rompers (torso length)                         | romper, rompers, cute, shorts, long, short, love, small, torso, flattering                                                                 |
| 20    | 86    | colors (as pictured)                           | pink, orange, red, color, mustard, green, got, online, looks, ordered                                                                      |
| 21    | 84    | shorts                                         | shorts, short, perfect, fit, length, great, legs, hard, love, comfortable                                                                  |
| 22    | 75    | soft fabric                                    | soft, super soft, super, fabric, soft comfortable, material, soft comfy, comfy, great, material soft                                       |
| 23    | 72    | coats                                          | coat, warm, trench, unlined, like, rain, medium, amp, winter, love                                                                         |
| 24    | 71    | leggings                                       | leggings, legging, warm, comfortable, black, price, tights, pair, feel, love                                                               |
| 25    | 70    | itchy wool / scratchy knits                    | itchy, sweater, wool, itch, cute, like, really, amp, long, quot                                                                            |
| 26    | 69    | xl sizing (matches photos)                     | xl, size xl, ordered, matches, size, true size, true, photos, xl fit, version                                                              |
| 27    | 52    | layering: needs a cami, sheer, staticky        | cami, need cami, cami underneath, need, wear cami, underneath, sheer, think, mesh, wear                                                    |
| 28    | 50    | comfy bottoms                                  | comfy, great, pair, 70s, love, super comfy, flattering, winter, comfortable, look great                                                    |
| 29    | 45    | blazers                                        | blazer, jacket, size, wear, look, small, casual, wardrobe, buttons, love                                                                   |
| 30    | 44    | white items (sheerness)                        | white, bought white, sheer, ordered white, bought, shirt white, shirt, ordered, dryer, worn                                                |
| 31    | 39    | runs large / runs small                        | runs, runs large, large, does run, big, run, runs little, little, reviews, say runs                                                        |
| 32    | 39    | looks different on the hanger than on the body | hanger, dress, tried, look, looked, looks, store, body, really, dresses                                                                    |
| 33    | 38    | medium / large fit reports                     | medium, large, ordered, medium fits, size, medium fit, fits, fit, med, ordered large                                                       |
| 34    | 37    | AG jeans (brand)                               | ag, jeans, ag jeans, stevie, pair, stretch, pairs, jeans fit, rise, fit                                                                    |
| 35    | 37    | black and neutral colorways                    | black, got black, bought black, bought, white, great, fits, got, blue, fabric                                                              |
| 36    | 36    | bras                                           | bra, bras, cups, straps, padding, like, racerback, support, bra underneath, cup                                                            |
| 37    | 35    | comfortable (generic)                          | comfortable, comfy, comfortable looks, comfortable great, cute comfortable, great, versatile, comfy great, ve owned, comfortable versatile |
| 38    | 31    | ponchos                                        | poncho, love, fall, center, oversized, underneath, looks, purchase, color, itchy                                                           |
| 39    | 30    | corduroy pants                                 | cords, grey, fit, pair, rise, color, navy, ag, 26, nice                                                                                    |
| 40    | 30    | care: dry-clean only, delicate, washing        | dry, washed, shrunk, wash, cold, cold water, washing, water, washed cold, instructions                                                     |
| 41    | 30    | fits true to size (tts)                        | tts, fit tts, fits tts, got, fit, model, navy, ordered, sizing, threads                                                                    |
| 42    | 28    | buttons: missing, falling off, too tight       | buttons, button, dress, missing, lost, holes, arrived, tried home, unbuttoning, website                                                    |
| 43    | 27    | true-to-size reports                           | size, true size, true, fits, fits true, large, size fits, perfect, love, recommend                                                         |
| 44    | 26    | swing dresses                                  | swing, dress, swing style, dresses, fabric, style, voluminous, version, like, moo                                                          |

## Topics by department

Top 5 topics per department, outlier documents excluded.

| department | topic | reviews | share_of_dept | top_words                                      | theme                                          |
|------------|-------|---------|---------------|------------------------------------------------|------------------------------------------------|
| Bottoms    | 4     | 517     | 30.9          | skirt, waist, skirts, size, great              | skirts and shorts (waist fit)                  |
| Bottoms    | 7     | 393     | 23.5          | jeans, pilcro, fit, stretch, 28                | jeans (sizing and stretch)                     |
| Bottoms    | 9     | 205     | 12.2          | pants, comfortable, look, love, fit            | pants                                          |
| Bottoms    | 18    | 111     | 6.6           | jumpsuit, jumper, love, torso, fit             | jumpsuits                                      |
| Bottoms    | 21    | 75      | 4.5           | shorts, short, perfect, fit, length            | shorts                                         |
| Dresses    | 0     | 2806    | 84.1          | dress, size, love dress, love, fit             | dresses (general)                              |
| Dresses    | 3     | 273     | 8.2           | xs, petite, xxs, dress, ordered                | xs / petite sizing (dresses, tops, sweaters)   |
| Dresses    | 13    | 42      | 1.3           | great, piece, fabric, quality, love            | quality, price and everyday basics             |
| Dresses    | 32    | 25      | 0.7           | hanger, dress, tried, look, looked             | looks different on the hanger than on the body |
| Dresses    | 44    | 23      | 0.7           | swing, dress, swing style, dresses, fabric     | swing dresses                                  |
| Intimate   | 16    | 126     | 19.0          | suit, swimsuit, bathing, bathing suit, bottoms | swimsuits                                      |
| Intimate   | 0     | 121     | 18.3          | dress, size, love dress, love, fit             | dresses (general)                              |
| Intimate   | 24    | 54      | 8.1           | leggings, legging, warm, comfortable, black    | leggings                                       |
| Intimate   | 12    | 49      | 7.4           | tank, great, love, wear, straps                | tanks                                          |
| Intimate   | 3     | 35      | 5.3           | xs, petite, xxs, dress, ordered                | xs / petite sizing (dresses, tops, sweaters)   |
| Jackets    | 11    | 174     | 39.3          | jacket, jackets, love, small, denim            | jackets                                        |
| Jackets    | 17    | 72      | 16.3          | vest, wear, love, soft, warm                   | vests                                          |
| Jackets    | 23    | 61      | 13.8          | coat, warm, trench, unlined, like              | coats                                          |
| Jackets    | 29    | 37      | 8.4           | blazer, jacket, size, wear, look               | blazers                                        |
| Jackets    | 0     | 23      | 5.2           | dress, size, love dress, love, fit             | dresses (general)                              |
| Tops       | 1     | 821     | 18.1          | shirt, love shirt, shirts, love, great         | shirts                                         |
| Tops       | 2     | 674     | 14.9          | sweater, love sweater, soft, color, love       | sweaters                                       |
| Tops       | 5     | 513     | 11.3          | love, great, cute, looks, soft                 | generic praise for tops                        |
| Tops       | 6     | 412     | 9.1           | blouse, size, blouses, wear, love              | blouses                                        |
| Tops       | 3     | 322     | 7.1           | xs, petite, xxs, dress, ordered                | xs / petite sizing (dresses, tops, sweaters)   |
| Trend      | 0     | 30      | 49.2          | dress, size, love dress, love, fit             | dresses (general)                              |
| Trend      | 4     | 9       | 14.8          | skirt, waist, skirts, size, great              | skirts and shorts (waist fit)                  |
| Trend      | 18    | 3       | 4.9           | jumpsuit, jumper, love, torso, fit             | jumpsuits                                      |
| Trend      | 11    | 2       | 3.3           | jacket, jackets, love, small, denim            | jackets                                        |
| Trend      | 2     | 2       | 3.3           | sweater, love sweater, soft, color, love       | sweaters                                       |
