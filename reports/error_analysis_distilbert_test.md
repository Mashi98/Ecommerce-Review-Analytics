# Error analysis — distilbert_test (test split)

3386 reviews, 586 misclassified. accuracy 0.8269, macro_f1 0.6670, weighted_f1 0.8379.

## Per-class

| class    | precision | recall | f1     | support |
|----------|-----------|--------|--------|---------|
| negative | 0.5848    | 0.6685 | 0.6239 | 356     |
| neutral  | 0.3967    | 0.5119 | 0.4470 | 420     |
| positive | 0.9631    | 0.8992 | 0.9301 | 2610    |

## Confusion matrix (rows true, columns predicted)

| true     | negative | neutral | positive |
|----------|----------|---------|----------|
| negative | 238      | 98      | 20       |
| neutral  | 135      | 215     | 70       |
| positive | 34       | 229     | 2347     |

## Error types

| error_type           | count | share_of_errors | share_of_class |
|----------------------|-------|-----------------|----------------|
| positive -> neutral  | 229   | 0.391           | 0.088          |
| neutral -> negative  | 135   | 0.230           | 0.321          |
| negative -> neutral  | 98    | 0.167           | 0.275          |
| neutral -> positive  | 70    | 0.119           | 0.167          |
| positive -> negative | 34    | 0.058           | 0.013          |
| negative -> positive | 20    | 0.034           | 0.056          |

## Error rate by star rating

| Rating | n    | errors | error_rate |
|--------|------|--------|------------|
| 1      | 123  | 25     | 0.203      |
| 2      | 233  | 93     | 0.399      |
| 3      | 420  | 205    | 0.488      |
| 4      | 733  | 208    | 0.284      |
| 5      | 1877 | 55     | 0.029      |

Labels come from ratings, so the 2-3 and 3-4 boundaries are where label noise is most likely.

## Error rate by review length (words)

| length | n    | errors | error_rate |
|--------|------|--------|------------|
| <=20   | 288  | 27     | 0.094      |
| 21-50  | 1082 | 164    | 0.152      |
| 51-100 | 1806 | 352    | 0.195      |
| >100   | 210  | 43     | 0.205      |

## Error rate by department (groups with at least 30 reviews)

| Department Name | n    | errors | error_rate |
|-----------------|------|--------|------------|
| Dresses         | 924  | 176    | 0.190      |
| Bottoms         | 570  | 106    | 0.186      |
| Jackets         | 136  | 23     | 0.169      |
| Tops            | 1469 | 238    | 0.162      |
| Intimate        | 268  | 40     | 0.149      |

## Error rate by class (groups with at least 30 reviews)

| Class Name | n   | errors | error_rate |
|------------|-----|--------|------------|
| Jeans      | 176 | 36     | 0.205      |
| Dresses    | 924 | 176    | 0.190      |
| Outerwear  | 44  | 8      | 0.182      |
| Skirts     | 129 | 23     | 0.178      |
| Shorts     | 45  | 8      | 0.178      |
| Pants      | 220 | 39     | 0.177      |
| Knits      | 725 | 120    | 0.166      |
| Blouses    | 421 | 69     | 0.164      |
| Jackets    | 92  | 15     | 0.163      |
| Lounge     | 134 | 21     | 0.157      |
| Fine gauge | 144 | 22     | 0.153      |
| Sweaters   | 179 | 27     | 0.151      |
| Swim       | 47  | 7      | 0.149      |
| Sleep      | 30  | 3      | 0.100      |

## Sample: positive -> neutral

| rating | true     | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 5      | positive | neutral   | Really beautiful pants but run a very large-i usually wear four and am 5 10-the media was way too big in the waste and elsewhere and very long.                                                                                                                                          |
| 5      | positive | neutral   | Based on the first person's review, i ordered a l at first... then i got it and it kinda fit like a maternity shirt and looked a bit too baggy/sloppy, not as flattering as the picture. i tried using the measurement guide below, based on the bust measurement, and i'm at the low e… |
| 4      | positive | neutral   | This was what i was looking for ! the little stripes ,black and white color,the shape of the jacket and skirt together....well perfect. kind of. i purchased this entire ensemble. the skirt was going to be shortened because well i'm short. no problem ,this is a common occurrence … |
| 5      | positive | neutral   | I loved the jumpsuit but unfortunately had to return it due to running a little on the small size.                                                                                                                                                                                       |
| 4      | positive | neutral   | I ordered this in a small and i am 32b and the fit is perfect. lovely details and very well made but the online pics make it look like a soft periwinkle blue, however when i pulled it out of the shipping bag, it was more gray-blue. dare i say...country blue. even the pictures i … |
| 4      | positive | neutral   | I love this dress in the photos, and like it enough to keep it--but i don't love it. it runs borderline true-to-size/slightly large. i'm 5'8", 133 lbs, 34 b and usually wear a 6 in dresses. i ordered a sm and med. am keeping the sm. medium was way too big. my main issue is the a… |
| 4      | positive | neutral   | I was searching for an ivory cardigan and i ordered this in my usual size small and an extra small. after trying them both on, the xs looked better, i like a more fitted look. a tad tight i the arms when layered but otherwise would have been a good fit, but the acrylic in this g… |
| 4      | positive | neutral   | This is a beautiful blouse that runs one to two sizes too small. i usually wear an 8, maybe a 10 on a bad day. i couldn't get into the 8, the 10 was uncomfortably tight, and even the 12 was snug (i haven't worn a size 12 anything since i was pregnant years ago). i think this blo… |

## Sample: neutral -> negative

| rating | true    | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|---------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3      | neutral | negative  | Fabulous fabric, adorable details. fitting is very, very odd. it's non-adjustable in any way. the fabric has no stretch at all around the middle and the underwire of the bra is very stiff which i found painful. the straps were kind of loose, but there is no way to tighten them b… |
| 3      | neutral | negative  | I wanted to love this dress because the color is so amazing, but the arm holes are huge! i am petite and tried this on and it just sadly didn't work.                                                                                                                                    |
| 3      | neutral | negative  | First, my husband had to help me zip up in this interesting but awkward design. the materials is nice, but stiff, so the skirt is quite full. the shoulder idea is interesting, but i would not want to spend an evening in it. to me, not up to the usual ms standard.                  |
| 3      | neutral | negative  | Has anybody washed this shirt a second time? just curious if it continues to shrink.                                                                                                                                                                                                     |
| 3      | neutral | negative  | I fell in love with this dress when the catalog came in the mail and jumped online immediately to order it. the dress just arrived and is not as pictured in the catalog - the picture of the model walking on the beach. in that picture it appears to have black lining or a black sl… |
| 3      | neutral | negative  | Fabric was a little more shear than i thought it would be - while not a big deal - it was missing a few of the silver beads along the front line of the top. returned                                                                                                                    |
| 3      | neutral | negative  | Thought this looked so cute, especially for averting frame. it arrived and was too large, the fabric felt super cheap, and the lace detail on the shoulders was very prominent.... and not in a good way. returned.                                                                      |
| 3      | neutral | negative  | Just a heads up - this is a romper - pants and not a dress! you can't tell from the pictures. i ordered thinking it was a dress but it's actually ... culottes? i'm not sure what you would call it. the pull on styling was awkward and the sash was bulky. it definitely goes back!    |

## Sample: negative -> neutral

| rating | true     | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2      | negative | neutral   | The color is a nice cool red and the material is very soft. but the dress left me feeling shapeless and the bell sleeve overtook it. i wanted to love it, but it just wasn't working for me. i'm 5'8" and purchased a s. it fits true to size, however it was loose around the waist an… |
| 2      | negative | neutral   | I wanted to love this dress, as the length, cut, and color are exactly what i've been looking for. but i was terribly disappointed in how thin and see through the material is.                                                                                                          |
| 2      | negative | neutral   | It was very pretty,but it ran very big , sadly i had to return it.                                                                                                                                                                                                                       |
| 2      | negative | neutral   | Looked really cute on the hanger, so i took it with me into the dressing room. the fabric is super soft, and the top is very flattering however the bottom is way too short! you can't tell from the picture but the sides are rounded and go up higher on your leg. the model has the … |
| 2      | negative | neutral   | When i received this dress in the mail, i was thrilled. i loved the color and the style, although a bit too much material but comfortable. when i went to show my husband the dress, his first comment was very sexy. i can see your underwear and your skin!!! i don't understand why … |
| 2      | negative | neutral   | If you are thinking about purchasing this dress without trying on in person, i bet you aren't going to be happy with it. for a phisique that is 5' 3" and 115lbs, this dress is not too short. however, the back key hole design is just not flattering. i felt like clothes in the bac… |
| 1      | negative | neutral   | Seemed pretty shapeless which surprised me because of the design. perhaps it was just too big. returned it.                                                                                                                                                                              |
| 1      | negative | neutral   | I have always bought interesting, pretty, wearable dresses for my daughters at retailer. now the dresses are just retro funky. sad to see.                                                                                                                                               |

## Disagreement with baseline_test (A = distilbert_test, B = baseline_test)

| outcome      | count |
|--------------|-------|
| both right   | 2478  |
| only A right | 322   |
| only B right | 207   |
| both wrong   | 379   |
