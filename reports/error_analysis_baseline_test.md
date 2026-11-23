# Error analysis — baseline_test (test split)

3386 reviews, 701 misclassified. accuracy 0.7930, macro_f1 0.6182, weighted_f1 0.8059.

## Per-class

| class    | precision | recall | f1     | support |
|----------|-----------|--------|--------|---------|
| negative | 0.4908    | 0.5983 | 0.5392 | 356     |
| neutral  | 0.3643    | 0.4667 | 0.4092 | 420     |
| positive | 0.9428    | 0.8720 | 0.9061 | 2610    |

## Confusion matrix (rows true, columns predicted)

| true     | negative | neutral | positive |
|----------|----------|---------|----------|
| negative | 213      | 98      | 45       |
| neutral  | 131      | 196     | 93       |
| positive | 90       | 244     | 2276     |

## Error types

| error_type           | count | share_of_errors | share_of_class |
|----------------------|-------|-----------------|----------------|
| positive -> neutral  | 244   | 0.348           | 0.093          |
| neutral -> negative  | 131   | 0.187           | 0.312          |
| negative -> neutral  | 98    | 0.140           | 0.275          |
| neutral -> positive  | 93    | 0.133           | 0.221          |
| positive -> negative | 90    | 0.128           | 0.034          |
| negative -> positive | 45    | 0.064           | 0.126          |

## Error rate by star rating

| Rating | n    | errors | error_rate |
|--------|------|--------|------------|
| 1      | 123  | 34     | 0.276      |
| 2      | 233  | 109    | 0.468      |
| 3      | 420  | 224    | 0.533      |
| 4      | 733  | 210    | 0.286      |
| 5      | 1877 | 124    | 0.066      |

Labels come from ratings, so the 2-3 and 3-4 boundaries are where label noise is most likely.

## Error rate by review length (words)

| length | n    | errors | error_rate |
|--------|------|--------|------------|
| <=20   | 288  | 28     | 0.097      |
| 21-50  | 1082 | 207    | 0.191      |
| 51-100 | 1806 | 409    | 0.226      |
| >100   | 210  | 57     | 0.271      |

## Error rate by department (groups with at least 30 reviews)

| Department Name | n    | errors | error_rate |
|-----------------|------|--------|------------|
| Tops            | 1469 | 313    | 0.213      |
| Dresses         | 924  | 192    | 0.208      |
| Bottoms         | 570  | 118    | 0.207      |
| Jackets         | 136  | 25     | 0.184      |
| Intimate        | 268  | 47     | 0.175      |

## Error rate by class (groups with at least 30 reviews)

| Class Name | n   | errors | error_rate |
|------------|-----|--------|------------|
| Skirts     | 129 | 33     | 0.256      |
| Knits      | 725 | 163    | 0.225      |
| Blouses    | 421 | 93     | 0.221      |
| Pants      | 220 | 47     | 0.214      |
| Lounge     | 134 | 28     | 0.209      |
| Dresses    | 924 | 192    | 0.208      |
| Fine gauge | 144 | 27     | 0.188      |
| Jeans      | 176 | 33     | 0.188      |
| Jackets    | 92  | 17     | 0.185      |
| Outerwear  | 44  | 8      | 0.182      |
| Sweaters   | 179 | 30     | 0.168      |
| Swim       | 47  | 7      | 0.149      |
| Shorts     | 45  | 5      | 0.111      |
| Sleep      | 30  | 3      | 0.100      |

## Sample: positive -> neutral

| rating | true     | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 4      | positive | neutral   | This is such a beautiful top. i sized down from a large to medium and it fits but too long and full for my shape. also found the armholes too deep and revealing for my large bust. sadly, returning.                                                                                    |
| 4      | positive | neutral   | Beautiful, unique design. it's very flattering the way the fabric hangs. i usually wear a 10, but because the lining is much smaller than the dress (which many linings are... it's a pet peeve of mine), i sized up to a 12 and took in the shoulders a couple inches so the bodice ar… |
| 5      | positive | neutral   | These are the perfect wide legs. they aren't overly wide (a common problem for petite women like myself, who cannot pull off the really wide styles), and they are nicely fitted in the top. the fabric is a great weight, neither too heavy nor too light so they lay nicely. (the sim… |
| 4      | positive | neutral   | This top is really cute for the right body shape. it is tight around the bust area if you select your size and sizing - you lose the shape.                                                                                                                                              |
| 4      | positive | neutral   | I ordered these in the taupe which is more like a rust color. they are really cute on, but they run a little big and long, so i had to return them. they would be great casual pants for the weekend though if the color and fit were better.                                            |
| 5      | positive | neutral   | These shorts were just what i expected from retailer. they are more pretty in person, particularly for the color - like a rich clay color, not garish orange. very good quality fabric & the embroidery around the edges is unique. i'm usually a size small in bottoms from retailer &… |
| 4      | positive | neutral   | Let me start off by saying that i really do like this sweater. the texture is very nice and, if it does start to pill, the material is one where i don't think it'll show up much (the fabric is a little "bumpy", but still soft). i'm only giving it 4 stars because the sleeves coul… |
| 4      | positive | neutral   | Super cute t-shirt keep in mind to is fairly long                                                                                                                                                                                                                                        |

## Sample: neutral -> negative

| rating | true    | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|---------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3      | neutral | negative  | I really wanted this to work. the style and colors are great and it is comfortable too. i didn't like how big the skirt portion was or how stiff the fabric was. it looked like a plaid tent on me but not in a terrible way. it could work on the right body.                           |
| 3      | neutral | negative  | I originally bought this shirt in the blood orange color, which is lovely and vibrant. the shirt is extremely comfortable and very flattering. however, it is also very, very cheap. after about five washes, it has pilled and worn, the color has faded a bit, and i doubt it will ma… |
| 3      | neutral | negative  | This is a cute dress, but the buttons are tiny and the top one cracked in half before i could even put the dress on. i can't risk the buttons breaking off while i am wearing it. especially since the dress seems designed for an a cup, anything larger will be more than those littl… |
| 3      | neutral | negative  | This dress was sooo cute online and in the store - i hands down would have bought it but i had an extra few minutes so i decided to try it on- terrible! the dress looked awakward on me- the strings lacing strings had town on a few spots and it just was not flattering for a curvy… |
| 3      | neutral | negative  | I tried on this shirt in the small and the xxs - both were ridiculously wide. they were also both cut really short - the peplum basically started right below my boobs, though the small was maybe a little bit longer than the eex. the stripes are really cute, the cut is just very … |
| 3      | neutral | negative  | I liked the idea of the diagonal color block but the fit of this tee was odd on me. too tight around arms and way big and boxy around the mid-section. going back.                                                                                                                       |
| 3      | neutral | negative  | The fabric is very thin. i have seen this comment on several sundry reviews, but its hard to imagine how thin the fabric is until you see it in person. i could put a cami under it - but it was also rather short on me. i don't usually have trouble with shirts running short and ca… |
| 3      | neutral | negative  | The blouse looked better on the model. i liked the way it fit, just not how it looks. the fabric made me itch, which was one of the other reasons i returned it. i wished it had looked good and the fabric didn't make me itch. i really liked the unique look to the shirt.            |

## Sample: negative -> neutral

| rating | true     | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2      | negative | neutral   | I absolutely love mother jeans, but i unfortunately had to return these ones. this particular style seemed to run pretty small. too, i always size up in white jeans, and even the size up that i had ordered were just too tight. i typically wear a 28 in mother, and ordered a 29 in… |
| 1      | negative | neutral   | I am not in love with this top and will be returning it. from the front it looks like a basic tee (which i expected) but the back "x" is too loose and slumps to the side so that it doesn't look like an "x" as the picture shows. so basically the focal point of the tee falls short. |
| 2      | negative | neutral   | I ordered this coat because, like others, i was taken by its feminine beauty! unfortunately, the coat showed up with the top button broken off (and missing) and another button hole torn open. i returned that coat right away, thinking that it'd been damaged somehow in shipping. h… |
| 1      | negative | neutral   | I saw this very shirt online & thought it was so pretty. in person, the shirt seems flimsy. the buttons look as though they'd fall off. i also didn't like the cut of the shirt. it seemed to be tighter on top & looser on the bottom. had to return. i couldn't justify the cost with… |
| 2      | negative | neutral   | This jacket looked adorable on. it was also quite comfortable. however, it sheds and left fuzz on every dark surface it touched. sadly, this is a deal breaker for me.                                                                                                                   |
| 2      | negative | neutral   | The pattern and fabric are lovely, but the sleeves are short (unlike the photo) so it drapes very strangely. i personally could not figure out how to wear it so that it looked anything remotely like the photo. had to return this one.                                                |
| 2      | negative | neutral   | I'm normally a size 25p. i bought these in a 26 because i wanted a more relaxed fit. i could barely pull them up over my calves. i could get them on and button them, but i felt like a sausage stuffed into a straw. size up 2 sizes at least!                                          |
| 2      | negative | neutral   | A beautiful shawl but not practical. i was impressed with how elegant/romantic it first felt/looked. however, after 5 minutes of wear, it would not stay on my shoulders. the design is very awkward and despite several attempts with different tops/tanks underneath, i kept running … |
