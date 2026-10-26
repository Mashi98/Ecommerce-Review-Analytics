# Error analysis — baseline (validation split)

3417 reviews, 700 misclassified. accuracy 0.7951, macro_f1 0.6245, weighted_f1 0.8083.

## Per-class

| class    | precision | recall | f1     | support |
|----------|-----------|--------|--------|---------|
| negative | 0.5140    | 0.6173 | 0.5609 | 358     |
| neutral  | 0.3586    | 0.4671 | 0.4057 | 426     |
| positive | 0.9445    | 0.8724 | 0.9070 | 2633    |

## Confusion matrix (rows true, columns predicted)

| true     | negative | neutral | positive |
|----------|----------|---------|----------|
| negative | 221      | 102     | 35       |
| neutral  | 127      | 199     | 100      |
| positive | 82       | 254     | 2297     |

## Error types

| error_type           | count | share_of_errors | share_of_class |
|----------------------|-------|-----------------|----------------|
| positive -> neutral  | 254   | 0.363           | 0.096          |
| neutral -> negative  | 127   | 0.181           | 0.298          |
| negative -> neutral  | 102   | 0.146           | 0.285          |
| neutral -> positive  | 100   | 0.143           | 0.235          |
| positive -> negative | 82    | 0.117           | 0.031          |
| negative -> positive | 35    | 0.050           | 0.098          |

## Error rate by star rating

| Rating | n    | errors | error_rate |
|--------|------|--------|------------|
| 1      | 124  | 38     | 0.306      |
| 2      | 234  | 99     | 0.423      |
| 3      | 426  | 227    | 0.533      |
| 4      | 741  | 219    | 0.296      |
| 5      | 1892 | 117    | 0.062      |

Labels come from ratings, so the 2-3 and 3-4 boundaries are where label noise is most likely.

## Error rate by review length (words)

| length | n    | errors | error_rate |
|--------|------|--------|------------|
| <=20   | 327  | 50     | 0.153      |
| 21-50  | 1074 | 189    | 0.176      |
| 51-100 | 1782 | 399    | 0.224      |
| >100   | 234  | 62     | 0.265      |

## Error rate by department (groups with at least 30 reviews)

| Department Name | n    | errors | error_rate |
|-----------------|------|--------|------------|
| Dresses         | 960  | 207    | 0.216      |
| Intimate        | 246  | 53     | 0.215      |
| Jackets         | 169  | 34     | 0.201      |
| Tops            | 1487 | 298    | 0.200      |
| Bottoms         | 546  | 105    | 0.192      |

## Error rate by class (groups with at least 30 reviews)

| Class Name | n   | errors | error_rate |
|------------|-----|--------|------------|
| Swim       | 57  | 20     | 0.351      |
| Outerwear  | 58  | 15     | 0.259      |
| Blouses    | 441 | 104    | 0.236      |
| Pants      | 193 | 45     | 0.233      |
| Dresses    | 960 | 207    | 0.216      |
| Lounge     | 89  | 19     | 0.213      |
| Jeans      | 173 | 33     | 0.191      |
| Knits      | 698 | 132    | 0.189      |
| Sweaters   | 194 | 35     | 0.180      |
| Fine gauge | 154 | 27     | 0.175      |
| Jackets    | 111 | 19     | 0.171      |
| Skirts     | 126 | 20     | 0.159      |
| Shorts     | 54  | 7      | 0.130      |
| Intimates  | 32  | 3      | 0.094      |

## Confident errors (probability >= 0.8)

29 errors (4% of all errors). The likeliest label-noise cases — read these first.

| rating | true     | predicted | conf | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3      | neutral  | positive  | 0.97 | I love this dress now that it is on sale; sized down to sm petite, added a simple full slip. endless compliments and inquiries on where to buy it!                                                                                                                                       |
| 3      | neutral  | negative  | 0.95 | If you like looking like a giant square and hide things this is the way to go. for me, it completely irradiates my shape, including my neck. this will be returned.                                                                                                                      |
| 4      | positive | neutral   | 0.92 | I really really like this tunic. it is gorgeous! but it runs small. i tried in l, usually wear l/xl, i may need to try the xl. but i sent it back because it just too expensive. the fabric itself is thick and heavy. i didn't like the fabric much, i wish it were softer. especially… |
| 4      | positive | neutral   | 0.90 | I so wanted to love this little dress, but it just didn't fit my body well. as suggested by others, i ordered a size up, and it did fit. the fabric is stretchy, so there is some give there, but the size zip was a little sticky, because the trim causes a lot of fabric to be bunch… |
| 3      | neutral  | negative  | 0.90 | I really like the sundry brand. however. this is thin and to me out of character for an retailer tee. it look adorable in picture. however the coloring is way off. the navy is almost black it's so dark and the kaki stripe is a green. it's nothing like i had hoped it would be. fa… |
| 3      | neutral  | negative  | 0.89 | This was so large on me that it looked awful. huge armholes and a baggy mess. material was nice and i loved the design. not sure a smaller size would even help.                                                                                                                         |
| 2      | negative | neutral   | 0.88 | This sweater is so cute and seems high quality. i would love to keep it but it is way too cropped for my taste, going down just below my mid section. i wish it had been cut longer- i would have kept it.                                                                               |
| 3      | neutral  | positive  | 0.88 | Nice dress to wear while seasons are transitioning                                                                                                                                                                                                                                       |
| 1      | negative | positive  | 0.88 | I saw this dress in the catalog and had to have it. when i went to the store it was sold out so i had them order and ship it to me in two sizes - my regular and one size larger. when i got them, i tried on my regular size first and it was just perfect! beautiful pattern and mate… |
| 3      | neutral  | positive  | 0.87 | Love the feel and comfort                                                                                                                                                                                                                                                                |
| 5      | positive | neutral   | 0.87 | Adorable soft top for the fall! i wanted to love it because of the colors and the fact that it is so retro! it was just way too big for me. i'm always a medium but in this i probably should have sized down. i still gave it 5 stars because it is so cute--just didn't fit me! sadly… |
| 3      | neutral  | positive  | 0.86 | I loved the material's texture and drape, but it was too sheer for me. i am 100% against having to wear a tank under white tees. (which means i have a lot of trouble finding any white tees, but...) other than the see-through nature, i loved this shirt!                             |
| 3      | neutral  | positive  | 0.85 | I bought this sweater in dark grey in the store. i really loved it at first - it's very soft and cozy and fits really well. i especially like that it's long enough to cover my tummy and butt when i wear skinny jeans or leggings. however, the other reviews that mention it gets fu… |
| 4      | positive | neutral   | 0.85 | This is an adorable dress that just didn't work for me! so sad! i wanted to love it because the pattern is super cute. i gave it 4 stars because it is very pretty and i can see it working on others!                                                                                   |
| 3      | neutral  | positive  | 0.85 | I tried this in my normal sp. i found it to run a little snug, but the neckline was way too low for every-day wear.                                                                                                                                                                      |

## Sample: positive -> neutral

| rating | true     | predicted | conf | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 4      | positive | neutral   | 0.51 | I am still in love with my ribboned denim dress from earlier this year. so, when i saw this dress i was super excited and ordered it right away in the same size. it just came and... i'm kinda disappointed. the fit is... shrunken?? kind of like i'm wearing a petite size. it's aro… |
| 5      | positive | neutral   | 0.38 | Was scared at first to order this product, when i tried it on i was amazed on how good it looks. fits just like the model picture, and the material is awesome.                                                                                                                          |
| 5      | positive | neutral   | 0.67 | If you know one.september and you know what's good for you, then size the heck down. maybe even two sizes if your particularly slim or want the dress to fit tighter. any way, this dress is simply scrumptious! the embroidery at the bodice is so detailed and pretty. the layers are… |
| 4      | positive | neutral   | 0.58 | This is a beautiful blouse however it runs big. i order the small and had to return it. my measurements are 5'4 135 34c-28-38 for reference. you will not regret buying this blouse but it has to be the right size if not it would look like it did on me, frumpy. i kept taking it of… |
| 5      | positive | neutral   | 0.64 | This is a beautiful dress. the dress is a lighter blue when compared to the picture, and the fabric is thick but the shape is beautiful. the waist is very fitted but it didn't feel uncomfortable. i didn't think the fabric was itchy as others mentioned. i do think it is a bit sho… |
| 5      | positive | neutral   | 0.53 | Tried this is my normal 4, and it fits exactly as shown. i can see how this could give issues to someone with any curves because the lining is definitely more fitted than the loose overlay, especially in the bust and hips. i found the quality to be very nice, length was slightly… |
| 4      | positive | neutral   | 0.53 | The material and design of this beach dress is so pretty. i was hoping that this could be worn as both a dress and a swim suit cover-up, but it is only the latter. you can't see it from the photos (the model or mine), but it is very sheer.... it can really only be worn over a sw… |
| 4      | positive | neutral   | 0.47 | I ordered this on-line, in a size 8. the fit is ample and fine fine, and the fabric is acceptable, but as much as i wanted to like it, and keep it, i ultimately returned it. it is just too, too conservative for me. i loved the deep olive/khaki colour, and even experimented weari… |

## Sample: neutral -> negative

| rating | true    | predicted | conf | review                                                                                                                                                                                                                                                                                   |
|--------|---------|-----------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 3      | neutral | negative  | 0.76 | It looks great on the model but not on me. on me, it looked more like a sack of potatoes. the fabric is unexpected for the style in my opinion. it is very thick. i'll be returning it because it is not flattering on me.                                                               |
| 3      | neutral | negative  | 0.45 | The style of this dress is quite lovely, i immediately gravitated towards it when i saw in the store, however once i tried it on it looked like a sack of potatoes...unfortunately the loose silhouette did not do anything for my figure, i should mention that i am 5'9" and have an … |
| 3      | neutral | negative  | 0.55 | Colors washed me out, i need more vibrant colors. so cute on the model                                                                                                                                                                                                                   |
| 3      | neutral | negative  | 0.47 | This is not worth $128. it looks cute on but....there is no support in the bust area. just a slight tuck in the fabric -it also is very flowy which is great and it is comfortable but for the price and the thinness and lack of meatiness to the fabric itself i cannot recommend and… |
| 3      | neutral | negative  | 0.68 | I wanted to love this top to go with a pair of red flats i also bought at retailer. but it's almost like a peplum crop top because the cut is short. the peplum piece also flares out rather dramatically. had to return it even though i love other tops from maeve.                    |
| 3      | neutral | negative  | 0.41 | Beautiful color; however, the fabric did not work for me. tight on top, large on bottom.                                                                                                                                                                                                 |
| 3      | neutral | negative  | 0.71 | Looks great from the front, but from the side it looks like a maternity top.                                                                                                                                                                                                             |
| 3      | neutral | negative  | 0.44 | I love love love the soft fabric and the cozy cowl. however mine not only had a giant hole, it seemed like the yarn started pilling as soon as i touched it. love the idea, but not practical, even without a hole =)                                                                    |

## Sample: negative -> neutral

| rating | true     | predicted | conf | review                                                                                                                                                                                                                                                                                   |
|--------|----------|-----------|------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2      | negative | neutral   | 0.72 | As others have said, this is cropped, so will look funny with most pants that aren't high waisted. i'm going to try it with a few dresses before i return it. the lining is adorable, so i don't want to have to send it back.                                                           |
| 2      | negative | neutral   | 0.57 | I'm so upset that this didn't work out. i'm petite at 5'2 and ordered this skirt in petite, however, i felt like it swallowed me up. the fabric is thick and warm so this is definitely only cold weather attire. i'm the picture it looks like there is a little leg showing, but on m… |
| 2      | negative | neutral   | 0.54 | The fit of this top is nothing like in the photo. the blouse on the model must be pinned back to achieve that fitted silhouette. in reality this top is baggy and boxy and runs at least one size large. the embroidery, details, and soft material are great, but the fit is off entir… |
| 2      | negative | neutral   | 0.45 | It just didn't work for me. i don't know if it is my fair complexion and untanned legs, but this dress was a strike out. the orange floral design and beading is beautiful and the fitting i would say is pretty tts. for reference, i am 5'7" and 145lbs with 36" 30" 40" measurements… |
| 1      | negative | neutral   | 0.58 | I ordered this in a size 1 which i thought would be a small and it was very, very wide. it is very short too. i typically wear a small and am 5'8" and 132 lbs. it's going back unfortunately. very soft material though.                                                                |
| 2      | negative | neutral   | 0.54 | This sweater was pretty short. beyond that, when i raised my arms, the thickness of the sweater caused the shoulders to rise up and stay there when i put my arms down.                                                                                                                  |
| 2      | negative | neutral   | 0.51 | I was so excited to get this dress. i need it for a wedding, and i thought i had hit the jackpot. the dress arrived with a small tear. the seams are puckered and badly seen. for an almost $300.00 dress, the quality is terrible. i expect better from retailer. i'm so disappointed.  |
| 2      | negative | neutral   | 0.51 | This top looks so pretty on the model. however, the photo is deceiving as the bottom section is quite wide and loose. i am petite but between a c+ to d cup. the top is not really fitted, so my bust pushed the top out and made it look like a maternity piece. from the side it hung… |
