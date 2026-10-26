# Error analysis — distilbert (validation split)

3417 reviews, 580 misclassified. accuracy 0.8303, macro_f1 0.6823, weighted_f1 0.8426.

## Per-class

| class    | precision | recall | f1     | support |
|----------|-----------|--------|--------|---------|
| negative | 0.6211    | 0.6732 | 0.6461 | 358     |
| neutral  | 0.4054    | 0.5634 | 0.4715 | 426     |
| positive | 0.9668    | 0.8948 | 0.9294 | 2633    |

## Confusion matrix (rows true, columns predicted)

| true     | negative | neutral | positive |
|----------|----------|---------|----------|
| negative | 241      | 104     | 13       |
| neutral  | 118      | 240     | 68       |
| positive | 29       | 248     | 2356     |

## Error types

| error_type           | count | share_of_errors | share_of_class |
|----------------------|-------|-----------------|----------------|
| positive -> neutral  | 248   | 0.428           | 0.094          |
| neutral -> negative  | 118   | 0.203           | 0.277          |
| negative -> neutral  | 104   | 0.179           | 0.291          |
| neutral -> positive  | 68    | 0.117           | 0.160          |
| positive -> negative | 29    | 0.050           | 0.011          |
| negative -> positive | 13    | 0.022           | 0.036          |

## Error rate by star rating

| Rating | n    | errors | error_rate |
|--------|------|--------|------------|
| 1      | 124  | 24     | 0.194      |
| 2      | 234  | 93     | 0.397      |
| 3      | 426  | 186    | 0.437      |
| 4      | 741  | 213    | 0.287      |
| 5      | 1892 | 64     | 0.034      |

Labels come from ratings, so the 2-3 and 3-4 boundaries are where label noise is most likely.

## Error rate by review length (words)

| length | n    | errors | error_rate |
|--------|------|--------|------------|
| <=20   | 327  | 41     | 0.125      |
| 21-50  | 1074 | 161    | 0.150      |
| 51-100 | 1782 | 337    | 0.189      |
| >100   | 234  | 41     | 0.175      |

## Error rate by department (groups with at least 30 reviews)

| Department Name | n    | errors | error_rate |
|-----------------|------|--------|------------|
| Dresses         | 960  | 173    | 0.180      |
| Bottoms         | 546  | 97     | 0.178      |
| Tops            | 1487 | 249    | 0.167      |
| Jackets         | 169  | 28     | 0.166      |
| Intimate        | 246  | 31     | 0.126      |

## Error rate by class (groups with at least 30 reviews)

| Class Name | n   | errors | error_rate |
|------------|-----|--------|------------|
| Shorts     | 54  | 12     | 0.222      |
| Blouses    | 441 | 93     | 0.211      |
| Swim       | 57  | 12     | 0.211      |
| Outerwear  | 58  | 12     | 0.207      |
| Jeans      | 173 | 32     | 0.185      |
| Dresses    | 960 | 173    | 0.180      |
| Skirts     | 126 | 22     | 0.175      |
| Pants      | 193 | 31     | 0.161      |
| Knits      | 698 | 111    | 0.159      |
| Fine gauge | 154 | 24     | 0.156      |
| Jackets    | 111 | 16     | 0.144      |
| Lounge     | 89  | 11     | 0.124      |
| Sweaters   | 194 | 21     | 0.108      |
| Intimates  | 32  | 3      | 0.094      |

## Sample: positive -> neutral

| rating | true     | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 4      | positive | neutral   | I really really like this tunic. it is gorgeous! but it runs small. i tried in l, usually wear l/xl, i may need to try the xl. but i sent it back because it just too expensive. the fabric itself is thick and heavy. i didn't like the fabric much, i wish it were softer. especially… |
| 5      | positive | neutral   | I am returning the dress because it smells heavily of the last purchaser's deodorant and perfume. it was actually nauseating to put on. i am disappointed with retailer that i was sold this dress, full price, in this condition. otherwise, the dress is lovely. it is true to size. … |
| 4      | positive | neutral   | I'm keeping this dress but it's going straight to the dry cleaner because right out of the bag it has a very strong chemical smell... like it had been stored in a gas station. then, i'm replacing all the buttons. the fabric is a thick eyelet, folded over with these teeny, thin s… |
| 4      | positive | neutral   | I like this top, and while it doesn't look on me like it does the model, i think it will be good for work purposes. would say it runs big and does have the maternity look risk (i bought this two months postpartum, so i need tops that cover the post baby belly and this works for … |
| 4      | positive | neutral   | I really like this shirt. the design and colors are very good. i ordered the black one with cats, it was cool. however, in general i don't like high-low tops because they don't look flattering and you can see that it is asymmetrical from the front, makes it awkward. the shirt's … |
| 4      | positive | neutral   | I love the feel and design of this pullover. it is extremely soft and comfy to wear. however, i did find that it runs slightly large, is a bit boxy and the sleeves are a bit lengthy. probably not the best for a petite frame. i am trying to decide if i should keep it though simpl… |
| 4      | positive | neutral   | Like the other reviewers have said, this is a wider shirt, but the width done is in a stylish way. i did not think it looked frumpy or sack-like on. however, i found it to be rather short. i bought a 0 regular (not petite), and it fits shorter than many shirts i have from retail… |
| 4      | positive | neutral   | As you can tell from the pictures, it's nothing fancy; i think this dress will get a lot of wear for everyday living. simple and comfy. reviewers said it ran pretty small so i sized up but wished i hadn't. it is not very "fit                                                        |

## Sample: neutral -> negative

| rating | true    | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|---------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3      | neutral | negative  | I was so ready to love this dress but it runs extremely small - like you should maybe purchase a size 2 sizes bigger than what you normally wear.                                                                                                                                        |
| 3      | neutral | negative  | This dress looks so beautiful on the model and i couldn't resist ordering it on-line. i have an athletic build -- read small hips -- straight. i'm 5'5' and 135 lbs. i was very disappointed when i put on. it's made for a curvy build. the dress was just too big, especially around … |
| 3      | neutral | negative  | This is a very cute designed shirt but it fits very boxy and runs large                                                                                                                                                                                                                  |
| 3      | neutral | negative  | I was very excited to try out this dress. i've been looking for a velvet dress for a couple of years now, and.. i'm not digging this one. the fabric itself was a let down right off the bat. it's cheap looking to me. i've seen nice velvet and this is very shiny, thin, and slightl… |
| 3      | neutral | negative  | This top runs 2 full sizes large. had to send back and didn?t want to bother with getting a new one.                                                                                                                                                                                     |
| 3      | neutral | negative  | L space is frustrating. i wear a 0/2 top but with 32d's i tried a large in this bikini top because from past experience with l space they run small. there's just zero support. the tie design helps secure the girls in place, but also flattens them out if you tie it tight enough s… |
| 3      | neutral | negative  | I was disappointed in this bordeaux top, a brand i usually enjoy. i ordered a medium, my usual size in their tops. it was very very big and did not fit at all as pictured here. also, the fabric is somewhat sheer and while i tried it on wearing a darker bra, the combo of being wa… |
| 3      | neutral | negative  | I wanted to love this romper, but it just wasted right for me. i am 5'5", 135lbs, 34c, curvy/muscular frame and ordered size small. i may have liked it better with more room in the medium, decided to return. i still recommend trying this product, but it wasn't for me and my hips! |

## Sample: negative -> neutral

| rating | true     | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2      | negative | neutral   | I saw this in the store and was glad i didn't order it earlier online. i thought it would be beautifully stitched but it was rather flat and unimpressive in person. i like more unusual and flattering pieces, and this just wasn't it. no compliments here.                            |
| 2      | negative | neutral   | Runs big and looked unflattering. i am petite, might work on someone taller.                                                                                                                                                                                                             |
| 2      | negative | neutral   | I wanted to love this top. perfect weight, length, and shape. unfortunately, the seam goes right through the bust and just does not look right. if you are tiny on top, this will be lovely on you.                                                                                      |
| 2      | negative | neutral   | Had to return this for a couple of reasons: 1) was too small! it's made like one of those tiny mittens that stretches to fit any size hand. i didn't bother to order a size up because i wasn't sure whether to go from a small to a medium or to a large. 2) the main color isn't grey… |
| 2      | negative | neutral   | The design of these ankles pants is very cool and the rise is perfect. however, the fabric, despite 2% spandex, has absolutely no give from the knee down. the extensive stitching probably contributes to the lack of stretch. while i was able to pull them on (size 27 for 110 lb/5.… |
| 2      | negative | neutral   | Love the tegan tunic version of this dress. i assumed since it looked identical except for the length that the sizing was the same but i was wrong. this runs so big! i got a s in the tegan but the s in this dress looked like an oversized potato sack on. i highly recommend orderi… |
| 2      | negative | neutral   | This sweater looked so promising with the tie waist and pretty lacy edging, but turned out to be a disappointment. it was much shorter than it looks in the photos and barely came to my hips. and the plum color was much darker than i expected, more of a dull brownish-burgundy.     |
| 1      | negative | neutral   | This is so close to perfect, but there's a seam down the middle of the back and they didn't bother to match the polka dots. you can see the mismatch in the second photo. i was about to buy the jacket, and totally in love, when i turned and saw the error. i hoped it was one bad m… |

## Disagreement with baseline (A = distilbert, B = baseline)

| outcome      | count |
|--------------|-------|
| both right   | 2490  |
| only A right | 347   |
| only B right | 227   |
| both wrong   | 353   |
