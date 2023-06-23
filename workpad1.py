#/usr/bin/python

### Used as another workpad to execute functions from main.py

import numpy as np
from coords_papakolea import s1, s2, s4, s5, s13
from main import cel2ecef


h00 = cel2ecef(s1.time, s1.cel[0])
h01 = cel2ecef(s1.time, s1.cel[1])
h02 = cel2ecef(s1.time, s1.cel[2])
h03 = cel2ecef(s1.time, s1.cel[3])
h04 = cel2ecef(s1.time, s1.cel[4])
h05 = cel2ecef(s1.time, s1.cel[5])
h06 = cel2ecef(s1.time, s1.cel[6])
h07 = cel2ecef(s1.time, s1.cel[7])
h08 = cel2ecef(s1.time, s1.cel[8])
h09 = cel2ecef(s1.time, s1.cel[9])
h10 = cel2ecef(s1.time, s1.cel[10])
h11 = cel2ecef(s1.time, s1.cel[11])

h12 = cel2ecef(s2.time, s2.cel[0])
h13 = cel2ecef(s2.time, s2.cel[1])
h14 = cel2ecef(s2.time, s2.cel[2])
h15 = cel2ecef(s2.time, s2.cel[3])
h16 = cel2ecef(s2.time, s2.cel[4])
h17 = cel2ecef(s2.time, s2.cel[5])
h18 = cel2ecef(s2.time, s2.cel[6])
h19 = cel2ecef(s2.time, s2.cel[7])
h20 = cel2ecef(s2.time, s2.cel[8])
h21 = cel2ecef(s2.time, s2.cel[9])
h22 = cel2ecef(s2.time, s2.cel[10])
h23 = cel2ecef(s2.time, s2.cel[11])
h24 = cel2ecef(s2.time, s2.cel[12])
h25 = cel2ecef(s2.time, s2.cel[13])

h26 = cel2ecef(s4.time, s4.cel[0])
h27 = cel2ecef(s4.time, s4.cel[1])
h28 = cel2ecef(s4.time, s4.cel[2])
h29 = cel2ecef(s4.time, s4.cel[3])
h30 = cel2ecef(s4.time, s4.cel[4])
h31 = cel2ecef(s4.time, s4.cel[5])
h32 = cel2ecef(s4.time, s4.cel[6])
h33 = cel2ecef(s4.time, s4.cel[7])
h34 = cel2ecef(s4.time, s4.cel[8])
h35 = cel2ecef(s4.time, s4.cel[9])
h36 = cel2ecef(s4.time, s4.cel[10])
h37 = cel2ecef(s4.time, s4.cel[11])

h38 = cel2ecef(s5.time, s5.cel[0])
h39 = cel2ecef(s5.time, s5.cel[1])
h40 = cel2ecef(s5.time, s5.cel[2])
h41 = cel2ecef(s5.time, s5.cel[3])
h42 = cel2ecef(s5.time, s5.cel[4])
h43 = cel2ecef(s5.time, s5.cel[5])
h44 = cel2ecef(s5.time, s5.cel[6])
h45 = cel2ecef(s5.time, s5.cel[7])
h46 = cel2ecef(s5.time, s5.cel[8])
h47 = cel2ecef(s5.time, s5.cel[9])
h48 = cel2ecef(s5.time, s5.cel[10])
h49 = cel2ecef(s5.time, s5.cel[11])

h50 = cel2ecef(s13.time, s13.cel[0])
h51 = cel2ecef(s13.time, s13.cel[1])
h52 = cel2ecef(s13.time, s13.cel[2])
h53 = cel2ecef(s13.time, s13.cel[3])
h54 = cel2ecef(s13.time, s13.cel[4])
h55 = cel2ecef(s13.time, s13.cel[5])
h56 = cel2ecef(s13.time, s13.cel[6])
h57 = cel2ecef(s13.time, s13.cel[7])
h58 = cel2ecef(s13.time, s13.cel[8])
h59 = cel2ecef(s13.time, s13.cel[9])
h60 = cel2ecef(s13.time, s13.cel[10])

hoku = np.array([[-8.913847985220527681e-01, -3.912119500877490053e-01, 2.288806480948964994e-01], [-8.560664194715192910e-01, -4.707567893095252476e-01, 2.133971198779406053e-01]])




'''cel = np.array([[0.8457229733467102, -0.4829367399215698, 0.22699062526226044], [0.8885384798049927, -0.40719014406204224, 0.21141304075717926], [0.7925061583518982, -0.5319826602935791, 0.29820868372917175], [0.7262102365493774, -0.5981862545013428, 0.33880943059921265], [0.8056207895278931, -0.5125254988670349, 0.29714086651802063], [0.7676920890808105, -0.5488230586051941, 0.33082035183906555], [0.7968429327011108, -0.5000013113021851, 0.3391754925251007], [0.8314138650894165, -0.4445149898529053, 0.33340272307395935], [0.7266574501991272, -0.5578458905220032, 0.4009699523448944], [0.7680473327636719, -0.5080094933509827, 0.3899097740650177], [0.8200508952140808, -0.43676862120628357, 0.36979687213897705], [0.7487872838973999, -0.502003014087677, 0.4327939450740814], 
                [0.840255618095398, -0.5001983046531677, 0.20921795070171356], [0.8885384798049927, -0.40719014406204224, 0.21141304075717926], [0.7925061583518982, -0.5319826602935791, 0.29820868372917175], [0.7262102365493774, -0.5981862545013428, 0.33880943059921265], [0.8056207895278931, -0.5125254988670349, 0.29714086651802063], [0.7676920890808105, -0.5488230586051941, 0.33082035183906555], [0.8020831942558289, -0.49333488941192627, 0.3365756571292877], [0.8314138650894165, -0.4445149898529053, 0.33340272307395935], [0.7266574501991272, -0.5578458905220032, 0.4009699523448944], [0.7680473327636719, -0.5080094933509827, 0.3899097740650177], [0.7487872838973999, -0.502003014087677, 0.4327939450740814], 
                [0.7666085958480835, -0.6371735334396362, -0.07950566709041595], [0.793708324432373, -0.6005016565322876, -0.09708186984062195], [0.779152512550354, -0.6237573027610779, -0.062034137547016144], [0.8052520751953125, -0.5888493657112122, -0.06946558505296707], [0.8674281239509583, -0.4844226539134979, -0.11359211802482605], [0.8506584763526917, -0.520404577255249, -0.07456028461456299], [0.8725466132164001, -0.4870811700820923, -0.037608835846185684], [0.7177644371986389, -0.6896508932113647, 0.0958954468369484], [0.8243351578712463, -0.5656575560569763, 0.02243037149310112], [0.8174088597297668, -0.5747260451316833, 0.03915104269981384], [0.8761771321296692, -0.4817296266555786, -0.015816211700439453], [0.7510417699813843, -0.6538882851600647, 0.09146814793348312], [0.87835294008255, -0.4779801070690155, -0.00558199780061841], [0.8683311343193054, -0.49587252736091614, 0.010554568842053413], [0.83592289686203, -0.5468424558639526, 0.04686465859413147], [0.7686341404914856, -0.6285987496376038, 0.11859656870365143], [0.7627377510070801, -0.6339122653007507, 0.12800884246826172], [0.8113254904747009, -0.5758799910545349, 0.10056441277265549], [0.8207122683525085, -0.562703549861908, 0.0989750325679779], [0.7267689108848572, -0.663973331451416, 0.1759156584739685], [0.7389245629310608, -0.6509921550750732, 0.17378082871437073], 
                [0.779152512550354, -0.6237573027610779, -0.062034137547016144], [0.8052520751953125, -0.5888493657112122, -0.06946558505296707], [0.8674281239509583, -0.4844226539134979, -0.11359211802482605], [0.8506584763526917, -0.520404577255249, -0.07456028461456299], [0.8725466132164001, -0.4870811700820923, -0.037608835846185684], [0.8243351578712463, -0.5656575560569763, 0.02243037149310112], [0.8174088597297668, -0.5747260451316833, 0.03915104269981384], [0.8761771321296692, -0.4817296266555786, -0.015816211700439453], [0.87835294008255, -0.4779801070690155, -0.00558199780061841], [0.8683311343193054, -0.49587252736091614, 0.010554568842053413], [0.83592289686203, -0.5468424558639526, 0.04686465859413147], [0.9090694785118103, -0.4159404933452606, -0.024211514741182327], [0.7686341404914856, -0.6285987496376038, 0.11859656870365143], [0.7627377510070801, -0.6339122653007507, 0.12800884246826172], [0.8113254904747009, -0.5758799910545349, 0.10056441277265549], [0.8207122683525085, -0.562703549861908, 0.0989750325679779], [0.7267689108848572, -0.663973331451416, 0.1759156584739685], [0.7389245629310608, -0.6509921550750732, 0.17378082871437073], [0.9155327081680298, -0.4015246331691742, 0.024037716910243034], 
                [0.8725466132164001, -0.4870811700820923, -0.037608835846185684], [0.8243351578712463, -0.5656575560569763, 0.02243037149310112], [0.7510417699813843, -0.6538882851600647, 0.09146814793348312], [0.8174088597297668, -0.5747260451316833, 0.03915104269981384], [0.8761771321296692, -0.4817296266555786, -0.015816211700439453], [0.87835294008255, -0.4779801070690155, -0.00558199780061841], [0.8683311343193054, -0.49587252736091614, 0.010554568842053413], [0.83592289686203, -0.5468424558639526, 0.04686465859413147], [0.7686341404914856, -0.6285987496376038, 0.11859656870365143], [0.9090694785118103, -0.4159404933452606, -0.024211514741182327], [0.7627377510070801, -0.6339122653007507, 0.12800884246826172], [0.8113254904747009, -0.5758799910545349, 0.10056441277265549], [0.8207122683525085, -0.562703549861908, 0.0989750325679779], [0.921913743019104, -0.38739505410194397, -0.00035016611218452454], [0.9155327081680298, -0.4015246331691742, 0.024037716910243034], [0.874726414680481, -0.4765290915966034, 0.08816909044981003], [0.8822206854820251, -0.45829150080680847, 0.10796099901199341], [0.817200779914856, -0.5502461791038513, 0.17149938642978668], [0.9025260210037231, -0.4186539947986603, 0.10087385028600693], [0.9171726703643799, -0.389987975358963, 0.08187607675790787],
                [0.7510417699813843, -0.6538882851600647, 0.09146814793348312], [0.8683311343193054, -0.49587252736091614, 0.010554568842053413], [0.83592289686203, -0.5468424558639526, 0.04686465859413147], [0.7686341404914856, -0.6285987496376038, 0.11859656870365143], [0.8113254904747009, -0.5758799910545349, 0.10056441277265549], [0.8207122683525085, -0.562703549861908, 0.0989750325679779], [0.874726414680481, -0.4765290915966034, 0.08816909044981003], [0.8822206854820251, -0.45829150080680847, 0.10796099901199341], [0.817200779914856, -0.5502461791038513, 0.17149938642978668], [0.9025260210037231, -0.4186539947986603, 0.10087385028600693], [0.840255618095398, -0.5001983046531677, 0.20921795070171356],
                [0.7177644371986389, -0.6896508932113647, 0.0958954468369484], [0.8725466132164001, -0.4870811700820923, -0.037608835846185684], [0.8243351578712463, -0.5656575560569763, 0.02243037149310112], [0.8174088597297668, -0.5747260451316833, 0.03915104269981384], [0.8761771321296692, -0.4817296266555786, -0.015816211700439453], [0.87835294008255, -0.4779801070690155, -0.00558199780061841], [0.8683311343193054, -0.49587252736091614, 0.010554568842053413], [0.83592289686203, -0.5468424558639526, 0.04686465859413147], [0.7686341404914856, -0.6285987496376038, 0.11859656870365143], [0.9090694785118103, -0.4159404933452606, -0.024211514741182327], [0.7627377510070801, -0.6339122653007507, 0.12800884246826172], [0.8113254904747009, -0.5758799910545349, 0.10056441277265549], [0.8207122683525085, -0.562703549861908, 0.0989750325679779], [0.921913743019104, -0.38739505410194397, -0.00035016611218452454], [0.9155327081680298, -0.4015246331691742, 0.024037716910243034], [0.874726414680481, -0.4765290915966034, 0.08816909044981003], [0.8822206854820251, -0.45829150080680847, 0.10796099901199341], [0.817200779914856, -0.5502461791038513, 0.17149938642978668], [0.9025260210037231, -0.4186539947986603, 0.10087385028600693], [0.9171726703643799, -0.389987975358963, 0.08187607675790787],
                [0.8920862078666687, -0.4310940206050873, -0.13542570173740387], [0.8725466132164001, -0.4870811700820923, -0.037608835846185684], [0.8174088597297668, -0.5747260451316833, 0.03915104269981384], [0.87835294008255, -0.4779801070690155, -0.00558199780061841], [0.9090694785118103, -0.4159404933452606, -0.024211514741182327], [0.921913743019104, -0.38739505410194397, -0.00035016611218452454], [0.9155327081680298, -0.4015246331691742, 0.024037716910243034], [0.874726414680481, -0.4765290915966034, 0.08816909044981003], 
                [0.9004268050193787, -0.2519787549972534, 0.35459595918655396], [0.8699069619178772, -0.28882837295532227, 0.39980006217956543], [0.7998872399330139, -0.3673906624317169, 0.474557101726532], [0.8671442270278931, -0.2734024226665497, 0.4163075387477875], [0.7411457896232605, -0.38899514079093933, 0.5471615195274353], [0.7423821687698364, -0.386510968208313, 0.5472458004951477], [0.8216774463653564, -0.2919527292251587, 0.4894994795322418], [0.8788905143737793, -0.20649674534797668, 0.4300122857093811], [0.7346730828285217, -0.37112265825271606, 0.5679114460945129], [0.8157784342765808, -0.2848738133907318, 0.5033413171768188], [0.7786039113998413, -0.32238927483558655, 0.5383689403533936], [0.9019109606742859, -0.1564474105834961, 0.40259265899658203], [0.9072747230529785, -0.13811273872852325, 0.3972121775150299], [0.8558512926101685, -0.21429140865802765, 0.47074171900749207], [0.7106330990791321, -0.3467160761356354, 0.6121997833251953], [0.8498125076293945, -0.14669236540794373, 0.506260871887207], [0.838071882724762, -0.14056316018104553, 0.5271409153938293], [0.7288992404937744, -0.26796117424964905, 0.6300021409988403], 
                [0.8699069619178772, -0.28882837295532227, 0.39980006217956543], [0.7998872399330139, -0.3673906624317169, 0.474557101726532], [0.8671442270278931, -0.2734024226665497, 0.4163075387477875], [0.7423821687698364, -0.386510968208313, 0.5472458004951477], [0.8216774463653564, -0.2919527292251587, 0.4894994795322418], [0.8788905143737793, -0.20649674534797668, 0.4300122857093811], [0.7346730828285217, -0.37112265825271606, 0.5679114460945129], [0.8157784342765808, -0.2848738133907318, 0.5033413171768188], [0.9019109606742859, -0.1564474105834961, 0.40259265899658203], [0.7786039113998413, -0.32238927483558655, 0.5383689403533936], [0.9072747230529785, -0.13811273872852325, 0.3972121775150299], [0.8558512926101685, -0.21429140865802765, 0.47074171900749207], [0.8498125076293945, -0.14669236540794373, 0.506260871887207], [0.838071882724762, -0.14056316018104553, 0.5271409153938293], [0.7288992404937744, -0.26796117424964905, 0.6300021409988403], [0.8345550298690796, -0.12902052700519562, 0.5356038808822632]])'''