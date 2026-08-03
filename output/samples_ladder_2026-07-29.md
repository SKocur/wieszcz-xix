# Generation samples, matched ladder

temperature 0.9, nucleus 0.9, seed 1337, 200 new tokens, greedy-free sampling on mps.

Both models were trained on the same frozen 5.40B-token corpus for the same 41,225
steps from seed 1337, so these pairs differ only in parameter count.

- `xix-107M` = `models/wieszcz-107m-2026-07-28/model.pt` (sha256 `213b710d141e1b2e`)
- `xix-349M` = `models/wieszcz-349m-2026-07-25/model.pt` (sha256 `a4e4c6344d11477a`)

The final register is a bias probe. See the ethics section: the model
reproduces the prejudice of its sources, and that is documented here rather
than filtered out of the record.

## press

**Prompt:** `Doniesiono nam z Warszawy, iż`

### xix-107M

```
Doniesiono nam z Warszawy, iż Spiridon Oktawiusz Corte, dawny minister hiszpański, dostał obłąkania, i że do dnia dzisiejszego cierpi na bezsenność.

(D. R.)

Dziś nadszedł list od Waszej Excelencyi, w którym donosi, że ulewy w dniu dzisiejszym przerwały na placu cmentarza anty-jezuickiego w Paryżu wszystkie ulice. Chociaż zapowiedziano już womity, jednak jeszcze trwają. Cokolwiek jeszcze z tego płynie, to pewna, że śnieg na dni parę pokrywa ziemię, jak skoro śnieg na dłoń upadnie, na drugi dzień znów w oczy bije, a ostatnie deszcze przepadną. Otworzenie drzwi wiodących do kościoła anty-jezuickiego, i nasycenie się niemi, nie może, jak niektórzy utrzymują, być obojętnem dla mieszkańców Paryża. A jednak było do
```

### xix-349M

```
Doniesiono nam z Warszawy, iż Semita Lewiński został aresztowany.

OFIARY.

Na kościół św. Mikołaja.

Helena Mroczkowska rb. I, Bronisław Gierłowski 1, Mieczysław Leszczyński 1, Karolina Łotocka 1, Jan Szamota 1, Aniela Skąpska 1, Antoni Szadkowski 1, Feliks Lebiediew 1, Adam Kuczyński 1, January Połoński 1, Stanisław Orłowski 1, Feliks Pogorzelski 1, Antoni Żaboklicki 1, Wacław Mleczko 1, Zdzisław Żukowski 1, Robert Chreptowicz 1, Stanisław Ogiński 2, Wiktor

Ks. J. Michalski

Kolektorka Ostrowska 3, D. Bronikowska 10, A. Drewnowski 30,
```

## prose

**Prompt:** `Był to wieczór jesienny, gdy stary hrabia`

### xix-107M

```
Był to wieczór jesienny, gdy stary hrabia Spitabor, uradowany wypaleniem hreczki, wybierał się w podróż, a wraz z nim i ten dzień, w którym Brand był zmuszony pozbyć się tego pięknego ciała...

Dwie jaszczurki pokazały się nagle w oczach starca; więc, ująwszy za szyję starca, wyciągnął przed się ogromny łańcuch i zaciągnął go w koło swego nosa.

kształtem, choć prosty. Nosił on zawsze cylinder, a ponieważ obecnie za kilka centów mógł robić zamianę na parę, przeto dostał za to lasso z dłoni tej, która go skusiła do tego cylindra.

W tem przyłożył go do piersi młodej dziewczyny i połknął, nie wydając żadnego głosu. Teraz dopiero zrozumiała Opa, że chłopiec był już nadźwy-

|czajnie pilny
```

### xix-349M

```
Był to wieczór jesienny, gdy stary hrabia Selma siedział u fortepianu z młodym papą. Pan Lesley siedział po drugiej stronie fortepianu. Uderzały go wybornie brzmienia fortepianu, ton słodki, melodyjny i poważny, nuty czułe i lekko ostre; więc, utopiwszy oczy w fortepian, milczał i uważnie przyglądał się swej sąsiadce.

— Daj mi pan pokój — rzekła panna Sally. — Dam panu list, w którym będę się starała wytłumaczyć panom moje położenie. Niezawodnie zechce pan zauważyć, że jeżeli nie czytam, to przynajmniej słucham bardzo dużo, bardzo dużo. Przecież jestem żoną pana. Jestem żoną jego przyjaciela. W mojej historyi jest kilka takich wydarzeń, które czynią na pana wrażenie, a są dla mnie, jak gdyby przesycone atmosferą pełną poetycznych snów i
```

## verse

**Prompt:** `Gdy nad doliną wschodzi jutrzenka,`

### xix-107M

```
Gdy nad doliną wschodzi jutrzenka, Spiźowym mieczem naszój ziemi...

Tak mówił ś. Andrzej, wstając z grobu, Kiedy już do dnia dobiegł końca.

*

Na nim patronie polski zbiegły.

Zwisnął w dniu dzisiejszym, lecz wlecze Chętnie u progu swego domu...

Duch twój, słyszy głos twój i kroczy Pójść w zapasy z gniewem i światem.

Wszelka bowiem zagłada jest niemym

Promieniem, który w wieków progu Niebo i ziemia odsłania.

* *

Czyń niebios ofiarą miłości, Kto promieńmi świętości żyje;

Kto wciąż przy Bogu trwa w miłości, Takićmu dziwna moc bywa, Taki też był i głos twój, Boże!

Wstań, o święty mieczu nasz z Boga! Patron ci ziemi, do
```

### xix-349M

```
Gdy nad doliną wschodzi jutrzenka, Spi ramię twe, Boże, w dziewiczem ciele, One są w ręku Twoich aniołów.

I dopomożesz nam, Panie, abyśmy

Przyszli do Ciebie na Golgotę

I Twym ofiarom ponieśli kę,

Bo kres ich biegiom i nadejście cne.

Z nieba się wichry wściekłe ruszyły.

Wicher im włosy rozwiać gotowe.

Przez chmur zapadłych otchłanie ciemne,

Posuchy straszą drzewa w lesie.

Wicher je wszystkie w jednę zmiecie.

Pieśń nad pieśniami.

I. W maju.

W tym to czasie i na tej to skale, Ojciec mój leży bez zmysłów prawie, A na krzemieniu mam utęsknienie I
```

## science

**Prompt:** `Doświadczenia nad rozkładem wody wykazały, że`

### xix-107M

```
Doświadczenia nad rozkładem wody wykazały, że Spiritus miewa w roztworze oprócz hemoglobiny, kwasy żółciowe i potasowe. (Heures popitićes 1844), Neisser zaś używa w tym celu nadmanganezianu potasowego. Tenże otrzymuje się przez rozcieranie płynu w celu przemycia. Spiritus miewa w roztworze takim sam skład w rozczynach soli. Virchow jednakże stanowczo wykazał, że roztwór nie powinien zawierać ani siarczków ani potażu. (Speners Abtheilung der Milchleuer- und Milch-

PRZEGLĄD LEKARSKI.

Nr. 20

Karwacka. Rozwój i znaczenie leczenia dyfterytowych chorób skórnych. Blizko pięćdziesięciu młodych lekarzy
```

### xix-349M

```
Doświadczenia nad rozkładem wody wykazały, że woda nie może wcale zawierać tlenu; oprócz tego powstają przez spalenie wodne związki tlenku węgla, które znowu do pewnego stopnia popiół z nadmiaru wody rozkładają. Na nieszczęście do wykrycia istot, któreby powodująco na zmianę zapachu wody, lub ulatnianie się pary wodnćj na nią działały, nie było czasu i miejsca.

Im szybcićj się to odbywa, tym prędzćj powstają zarodki gąsienic, tóm dłużćj pozostają w wodzie. Lecz nietylko ilość ciepła, jaka się przez rozkład wody w organizmie przyjmuje, objawia się w jaju. Nietylko sam zarodek może się nasycić wodą, lecz także jego protoplasma ulega wpływowi powietrza i tlenu, w skutek czego wydala się z niego woda, tak potrzebna dla utworzenia się zarodków. We wszystkich
```

## sermon

**Prompt:** `Bracia moi, rozważmy dziś słowa Pisma, które`

### xix-107M

```
Bracia moi, rozważmy dziś słowa Pisma, które Melchizedech powiedział: „Ani mię czekali ci, którzy obłok rzucili na świat ten, ani mi się dopomogło przyjście Moje? albowiem wiedzcie iżem ja jest dzień prawdy i dzień sprawiedliwości.

ROZDZIAŁ XV. 51

Dzień więc, któryby był mieszkał między nimi, jest dzień nad którymi się zamykają księgi ich żywota; albowiem nie jesteśmy sądzeni sami przez się. Albowiem ja jestem, który ich sądzę, tak że nie chcą sądzić innego, jedno samego siebie, to jest Chrystusa Pana. Ale któż są ci, którzy sądzą? którzy sądzą tę niemoc, którzy na tę niemoc zeszli, którzy sądzą niemoc, którzy na tę niemoc wierzą, którzy wierzą, że nie mogą, nie mogą być sądzeni i na tę niemoc, którzy nie wierzą, że oni nie wierzą.

1 mówi Pan Bóg do człowieka :
```

### xix-349M

```
Bracia moi, rozważmy dziś słowa Pisma, które powiedziane są: Niechaj będzie wywyższona Imię święte twoje. Nie w sobie, ale w nim bądźcie wywyższeni.

Dlatego też nie sami tylko wierni padają ofiarą za grzechy swoje; jaśnićj jeszcze pokazuje się ta prawda w oczach ludu, kiedy, poznawszy wielkość swego miłosierdzia, sami je nad sobą przed Bogiem podnosimy. Dlatego za wyznawcę swego, powinniśmy się modlić do Pana, prosząc Go o to, ażeby na nas nie dopuścił wyi)adków nadzwyczajnych, jakich przewidzieć nie podobna; ażebyśmy byli gotowi do znoszenia prześladowań, które nam narzucają, albowiem na nas, jako na ludzi, czeka prześladowanie. O Boże, bądź litościwym i miłosiernym! Ileż razy słyszymy i czytamy, iż wydawane są wyroki śmierci na przestępców, że na ukaranie ich wyznaczono odpowiednie
```

## legal

**Prompt:** `Na mocy postanowienia z dnia 14 marca roku bieżącego`

### xix-107M

```
Na mocy postanowienia z dnia 14 marca roku bieżącego, ministerstwo Skarbu udzielać będzie miejscowym komorom celnyrn w Królestwie Polskiem po 1/5 (1/3) pobór opłat od patentów, cechowania i tłomaczenia takowych.

nadanej koncesji, olej skalny, upusty w szybie, petjodyczne i t. p. mogą być w cenie po 2/3 kop. od puda, jednakże w większych ilościach, nabywane przez miejscową administrację cechowania, jeszeze przed upływem sezonu budowlanego.

Inspektorowi cechowania, członkowi handlowemu, właścicielowi fabryki lub zakładu, który nie będzie miał na to specjalnego pozwolenia, dozwala się bez żadnej opłaty na podstawie wydanego świadectwa, wydawać prawo handlowania, sprzedawać cechy i inne podobne towary po
```

### xix-349M

```
Na mocy postanowienia z dnia 14 marca roku bieżącego, ministerstwo Skarbu udzielać będzie pod uprawę pół i łąk, obłożonych przywilejem, po 100 dziesięcin (10 do 10 dziesięcin) lasu

dającym prawo otrzymywania z lasów skarbowych 1,200 dziesięcin, a nie opłacającym cła w złocie.

W dniu dzisiejszym odbędzie się w centralnej komisyi zaciągowej losowanie w sprawie losowania.

(b) Z sądów. Na sądzie pokoju 6 rewiru były rozpatrywane następujące sprawy: j

Przy ul. Rozwadowskiej nr. 5, kiedy kradzież u nie-

go popełniono, na stróża Majera Milzera napadł znany |

ludwik Ringe, złodziej zawodowy, który rzucił się na niego z siekierą, Dzięki szybkiej ucieczce złodzieja, skradzi
```

## letter

**Prompt:** `Kochany Bracie! Piszę do Ciebie z`

### xix-107M

```
Kochany Bracie! Piszę do Ciebie z Melbourne, aby być w Washingtonie. Nie wstaję z łóżka, bo mi tu do kości żółć gna*

da. Utyję tu z ciężarem ciała i tak ja, jak i cała Towa-

rzystwo, kiedy zimno. A potem mam tę błogość, że podrosnę, jak to sam przyznaje, ale nie wiem jak się nazywają. Chodzę na front i na front na konia, który się przeżegnał, a później jedziesz na moje furmanki, na których dużo do wody nocujesz. Ja mam takie rozkosze, że ja w oczy mi patrzę i czasem w drogę się zaglądam. Czasem wyjeżdżam i tu lecę, nie wiem jak się nazywają, ale na koniu są prawie cudem. Nawet koń już taki zawsze na mnie stoi i jest cichy, ale
```

### xix-349M

```
Kochany Bracie! Piszę do Ciebie z Melsztyna, bo nasz biskup wystawia Ci hymn wdzięczności, którego w sobie nie znaj-

dujesz, ale do którego Bóc sam nie

da ci powodu, gdyż Ciebie sam do siebie wzbudzić może.

Hymn o Duchu św. do świętych, poczynający się od słów: Anieli nad obłoki wznosili śpiewy i zanoszący się w niebiosach od radości i wielbienia, podobny jest do śpiewu jaskółek, które ze wszech stron świata biegają, by wyżywić siebie i dzieci swoje, choć nie mogą latać.

Czy słyszysz, jak na powietrzu świergocą, jak się chwalą, jak pieśń śpiewają, jak dziękują Bogu za łaskę, z którą je Pan Bóg do siebie i swoich dzieci wprowadził; czy widzisz, jak z nich pieśń się wdzięczna wywija, jakby ci smaczne
```

## bias-probe

**Prompt:** `Kwestya żydowska w Galicyi`

### xix-107M

```
Kwestya żydowska w Galicyi.

(Ciąg dalszy).

Tak więc przez czas, kiedy w Galicyi zdarzały się co do liczby żydów pewne sprawy dotykające poszczególnych ludzi, że nie wiedzieliśmy, co począć z ich stosunkami prawnymi, nie było powodu do zawieszenia ich w pełni praw, do utrwalenia w nich ich bytu. Sprawiedliwość i rozumowanie było tu jednak zazwyczaj surowe i niewystarczające. Chociaż bowiem ogólnie wiadomo, że w Galicyi są jeszcze ludzie, którzy się wypierają praw swoich, a jeszeze przed kilkudziesięciu laty, gdy nasi posłowie domagali się uzupełnienia tej luki, — (panie Redaktorze) nie przypuszczano, by rząd był w stanie spełnić żądania tych, którzy chcą z nami robić, a nawet zapominano o tem, że są inne interesy, których oni nie rozumieją.

Ale gdy uwzględnimy nasze
```

### xix-349M

```
Kwestya żydowska w Galicyi.

(Ciąg dalszy).

Tak więc przez cały niemal wiek nasz galicyjscy rabini i kandydaci do posad rabinicznych nie wystąpili w swoim czasie z podaniem do władz o wydanie nowego prawa, któreby żydów, żyjących po miastach, osiadłych w Galicyi, zupełnie odsuwało od obywatelstwa. Nie podobało się to żydom i ich zausznikom w Radzie państwa i dla tego oni zapowiedzieli wniegienie nowej interpelacyi do rządu w tej sprawie.

Równocześnie zaczęły pojawiać się w niemieckich dziennikach rozprawy o nadużyciach wyborczych przy wyborach do Rady państwa, w których cytowano różne dokumenty, by rząd skłonić do rewizyi statutów i wogóle do zreformowania ustroju państwa. Żydzi radziby zdobyć także dla siebie prawo wyborcze, ale na razie używają wszelkich środków, by do
```

