# Baseline for translations

To set a baseline for French question answering with low resources, we use DeepL to translate from French to English. Here we analyze the quality of the French to English translations in the Simple Question Task. 

## BLEU

The BLEU score between the DeepL translation and the English original is 45.4. There are 104 human translations for Simple Questions.  For the 2014 newstest set for English-French, DeepL Translator achieves a BLEU score of 44.7.

## Manual Error Bucketing

| French to English | English Original | Same predicate and object? | Error Bucket | Notes |
|:- |:- |:- |:- |:- |
| to which film does hamilton deane owe credit for the story of the film | to what film did hamilton deane receive film story credits? | 1 |
| what region does j. harold grady identify with? | what is the region j. harold grady identifies with? | 1 |
| album created by andy williams | what's an album andy williams made | 1 |
| album recorded by francesco guccini | whats an album recorded by francesco guccini  | 1 |
| album recorded by rahzel | what is an album recorded by rahzel | 1 |
| artist of the record label roc-a-fella | who is an artist on the record label roc-a-fella records? | 0 | wrong object | roc-a-fella records vs roc-a-fella |
| other name for devon w. carbado's black ethnicity | what is another name for devon w. carbado's black ethnicity | 1 |
| song from the album "my father and my sound" | what's a song off of my father and my son | 1 |
| which county is fayetteville in? | which county is fayetteville located in? | 1 |
| in which east coast state is stockton located | which east coast state is stockton located in | 1 |
| in which country was killing spree filmed? | what country was killing spree filmed in | 1 |
| what famous actor is melyssa james the parent of | which famous actor is melyssa james the parent of | 1 |
| which album is from quiet storm | what album is quiet storm a song on | 1 |
| what country is the film rossini from? rossini! | what country is the film rossini! rossini! from | 0 | wrong object | rossini! rossini! vs rossini |
| example of a compilation album | what is an example of a compilation album | 1 |
| horror movie from netflix | what movie titles belong to the netflix genre horror | 1 |
| flórián kaló comes from which country | flórián kaló comes from what country | 1 |
| where hermann von mallinckrodt died | where did hermann von mallinckrodt pass away | 1 |
| where was the birthplace of nikos lambrou | where was nikos lambrou's birthplace | 1 |
| where was the discovery (12971)4054 t-3 made? | where was the discovery of (12971) 4054 t-3 made? | 1 |
| where the frank frank wars took place | where were the frisian–frankish wars fought at | 0 | wrong object | frank frank wars vs frisian–frankish wars |
| where is the town of gila county / winkelman | where is gila county / winkelman town located | 1 |
| music track from the hard candy album | what track would you find on hard candy | 1 |
| what is a christian contemporary music album? | what is a album of contemporary christian music | 1 |
| what is an album recorded by ane brun? | what is an album recorded by ane brun  | 1 |
| what is a studio album? | what is a studio album? | 1 |
| what is a memory book? | what is a memoir book? | 1 |
| what is a type of radio station? | what is a type of radio network? | 0 | wrong object | radio station vs radio network |
| what is a popular song by annie ross | what is a popular track by annie ross | 1 |
| what is 11664 kashiwagi orbit | what does 11664 kashiwagi orbit  | 1 |
| what is 6864 starkenburg | what is 6864 starkenburg | 1 |
| what has liberated the united nations | what released united nations  | 0 | wrong predicate | liberated vs released |
| which actor was born in wilmington, north carolina? | which actor was born in wilmington, north carolina? | 1 |
| which album is produced by midori goto | what album is produced by midori goto | 1 |
| which artist recorded the song jungle fever? | which artist recorded the track jungle fever? | 1 |
| which artist recorded the sun years | which artist recorded the sun years | 1 |
| which artist uses a drum kit | what artist used a drum kit | 1 |
| which biofluidic site contains 2-aminobenzoic acid? | which biofluid location contains 2-aminobenzoic acid? | 1 |
| what is the example of a compilation album? | what is an example of a compilation album? | 1 |
| what is the active ingredient in hp tray? | what is the active ingredient in bac hp? | 0 | wrong object | hp tray vs bac hp |
| what is the active ingredient of cefepime 20? | what is the active ingredient in cefepime 20 injection  | 1 |
| what is the object of the book going to war? | what is the subject matter of the book going to the wars | 0 | wrong object | war vs wars | 
| what is the release track of the album lovely madness | what is the release track on a lovely madness | 1 |
| what is the language of the television program manager wanted with anne burrell? | what is the language of the tv program chef wanted with anne burrell | 1 |
| what's the name of a christmas movie | what is the name of a christmas film | 1 |
| what's the name of a fantasy book | what is the name of a fantasy book | 1 |

6 translations have different objects 
1 translation has a different predicate
46 translations were manually graded

Based on an analysis of 104 translations, it seems that the baseline has little to improve. There does not exist a significant gap here.