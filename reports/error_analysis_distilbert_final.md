# Error analysis — distilbert_final (validation split)

3417 reviews, 585 misclassified. accuracy 0.8288, macro_f1 0.6785, weighted_f1 0.8411.

## Per-class

| class    | precision | recall | f1     | support |
|----------|-----------|--------|--------|---------|
| negative | 0.6055    | 0.6816 | 0.6413 | 358     |
| neutral  | 0.4034    | 0.5493 | 0.4652 | 426     |
| positive | 0.9671    | 0.8940 | 0.9291 | 2633    |

## Confusion matrix (rows true, columns predicted)

| true     | negative | neutral | positive |
|----------|----------|---------|----------|
| negative | 244      | 103     | 11       |
| neutral  | 123      | 234     | 69       |
| positive | 36       | 243     | 2354     |

## Error types

| error_type           | count | share_of_errors | share_of_class |
|----------------------|-------|-----------------|----------------|
| positive -> neutral  | 243   | 0.415           | 0.092          |
| neutral -> negative  | 123   | 0.210           | 0.289          |
| negative -> neutral  | 103   | 0.176           | 0.288          |
| neutral -> positive  | 69    | 0.118           | 0.162          |
| positive -> negative | 36    | 0.062           | 0.014          |
| negative -> positive | 11    | 0.019           | 0.031          |

## Error rate by star rating

| Rating | n    | errors | error_rate |
|--------|------|--------|------------|
| 1      | 124  | 26     | 0.210      |
| 2      | 234  | 88     | 0.376      |
| 3      | 426  | 192    | 0.451      |
| 4      | 741  | 214    | 0.289      |
| 5      | 1892 | 65     | 0.034      |

Labels come from ratings, so the 2-3 and 3-4 boundaries are where label noise is most likely.

## Error rate by review length (words)

| length | n    | errors | error_rate |
|--------|------|--------|------------|
| <=20   | 327  | 41     | 0.125      |
| 21-50  | 1074 | 159    | 0.148      |
| 51-100 | 1782 | 345    | 0.194      |
| >100   | 234  | 40     | 0.171      |

## Error rate by department (groups with at least 30 reviews)

| Department Name | n    | errors | error_rate |
|-----------------|------|--------|------------|
| Dresses         | 960  | 175    | 0.182      |
| Tops            | 1487 | 255    | 0.171      |
| Bottoms         | 546  | 93     | 0.170      |
| Jackets         | 169  | 27     | 0.160      |
| Intimate        | 246  | 32     | 0.130      |

## Error rate by class (groups with at least 30 reviews)

| Class Name | n   | errors | error_rate |
|------------|-----|--------|------------|
| Outerwear  | 58  | 13     | 0.224      |
| Shorts     | 54  | 12     | 0.222      |
| Swim       | 57  | 12     | 0.211      |
| Blouses    | 441 | 89     | 0.202      |
| Jeans      | 173 | 33     | 0.191      |
| Dresses    | 960 | 175    | 0.182      |
| Fine gauge | 154 | 27     | 0.175      |
| Skirts     | 126 | 21     | 0.167      |
| Knits      | 698 | 112    | 0.160      |
| Pants      | 193 | 27     | 0.140      |
| Sweaters   | 194 | 27     | 0.139      |
| Jackets    | 111 | 14     | 0.126      |
| Lounge     | 89  | 11     | 0.124      |
| Intimates  | 32  | 2      | 0.062      |

## Sample: positive -> neutral

| rating | true     | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 4      | positive | neutral   | This top is really cute and the cap sleeves are perfect -- what i didn't notice from the online photos is that the back of the neckline is also scalloped which is a really sweet detail! however, i thought this t was a little tight through the chest (and i'm only a b cup!) and th… |
| 5      | positive | neutral   | I fell in love with this dress from the moment i saw it on various mother's day weekend insta posts by retailer stores. i was worried that it would not be a great cut for me, as it's pretty straight up and down, and i wasn't sure the waist would hit at the right place, because i… |
| 4      | positive | neutral   | The style and fit are excellent, doesn't make you look frumpy or larger. just can't get into the pattern combo--the white is too stark and i feel like it was slapped on my chest like a dickey. i have the paquerette and that's a mixed bag of patterns that works really well. i'm n… |
| 4      | positive | neutral   | This is a really pretty print on a decent top. the only reason i returned is because it didn't look good with my skin tone and the fabric is thin. i'd planned to use it as a later under cardigans.                                                                                     |
| 4      | positive | neutral   | I have been on the hunt for a replacement cargo jacket for about a year now. i thought this one was going to be it!! ordered it right away. i got it in an xl as retailer coats always seem to run small on me (i am very busty so sometimes the more fitted styles won't even close wi… |
| 4      | positive | neutral   | Beautiful top but runs a whole size small. gave to my daughter who loved!                                                                                                                                                                                                                |
| 4      | positive | neutral   | Lovely, lovely dress, great quality. unfortunately runs rather large at the hips. i wear a size 2 and the xs fit well in the chest but large at the hips.                                                                                                                                |
| 4      | positive | neutral   | It's a cute top, but the only complaint i have is that it is difficult for me take off. it goes on easily, but i get always have a bit of a struggle trying to take it off. my shoulders tend to get stuck so i can't pull my arms out easily. i just wish the top had an extra button … |

## Sample: neutral -> negative

| rating | true    | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|---------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3      | neutral | negative  | Tried on the xxs petite and still felt that it looked too boxy and baggy on me. cute concept but poor execution                                                                                                                                                                          |
| 3      | neutral | negative  | I got this dress in the mail, and while i loved the concept of it, it really is too boxy. the tie is very superficial and does not really nip you in at the waist, it is more so for show. it is fully lined which was interesting, and to me, made the dress seem boxier. the quality … |
| 3      | neutral | negative  | This skirt runs very large--even in the petite line. there was also no shape to the skirt (at least as it looked on my body). the color was nice (in the pink-grapeish hue), but ultimately the skirt just looked like a tent on my body (00p).                                          |
| 3      | neutral | negative  | Purchased because i wanted a deep red sweater, and this one was the perfect color. unfortunately i am on the curvier side, with 34g chest, 155lbs 5' 5", and the med was too big, (i usually wear m or l at retailer.) also the fit ended up being boxy on my body type.                 |
| 3      | neutral | negative  | This is a very cute designed shirt but it fits very boxy and runs large                                                                                                                                                                                                                  |
| 3      | neutral | negative  | This was my favorite shirt before i ordered it .... i absolutely couldn't wait to wear it. but i was so disappointed. at 5'2" and 120 i have a athletic frame ... the small was tight in the shoulders but had so much extra fabric that i felt lost. i wasn't wearing the shirt .... i… |
| 3      | neutral | negative  | I also wanted to love this top but sadly had to return it because it is very short and the peplum piece is very wide. it was not flattering at all.                                                                                                                                      |
| 3      | neutral | negative  | I had high hopes for this top, as it looked great online. unfortunately it was much more ruffly in person and not my style at all. it was a lot of fabric and poofed out too much -- reminded me of a child's top. sad to return it!                                                     |

## Sample: negative -> neutral

| rating | true     | predicted | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2      | negative | neutral   | I really wanted to love this waistcoat. i purchased this online and tried it on as soon as i received it. it runs large, but i should have realized the horizontal stripes sitting on the hips is not a good thing. the fur adds additional bulk where it is not needed, the hip area !… |
| 2      | negative | neutral   | I thought this skirt would fit just over the knee with a nice flowy build. instead it fit rather baggy and was longer than i expected. i felt dowdy in it.                                                                                                                               |
| 2      | negative | neutral   | The design is gorgeous, the fit is great and the fabric drapes beautifully and is very comfortable. however, it's extremely see thru.                                                                                                                                                    |
| 2      | negative | neutral   | The design of these ankles pants is very cool and the rise is perfect. however, the fabric, despite 2% spandex, has absolutely no give from the knee down. the extensive stitching probably contributes to the lack of stretch. while i was able to pull them on (size 27 for 110 lb/5.… |
| 2      | negative | neutral   | It looks adorable in the picture... but just not flattering in real life. too much fabric in the skirt that flares from the waist in a bunchy, unflattering way. makes a slender but curvy figure look boxy, rather than enhancing curves. it's a pass. love the blue print, though.     |
| 2      | negative | neutral   | The print and fabric are great, but this dress has odd construction, or maybe i have an odd construction. anyway, i'm 5-5, weigh 125-130, 36d. i ordered the small and it fit perfect up top and in the arms - but the issue was in the hips. i'm somewhere between athletic and curvy.… |
| 2      | negative | neutral   | I love the design, that's why i'm going back for a smaller size. i bought a medium and that size was made for someone 50 lbs heavier than i. usually i'm good at choosing my size online, but not today. if you are curvy and round like me still go for a smaller size because this dr… |
| 1      | negative | neutral   | This is a beautiful shirt. the back is a few inches longer than the front which makes for a cool effect. the only thing that i didn't like about this shirt was how fragile the buttons on the back are. i was taking this shirt off and one popped off. when i went to hang it, anothe… |

## Disagreement with baseline (A = distilbert_final, B = baseline)

| outcome      | count |
|--------------|-------|
| both right   | 2488  |
| only A right | 344   |
| only B right | 229   |
| both wrong   | 356   |
