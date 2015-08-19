Zaganjanje algoritma zna povzrociti kar nekaj tezav, zato vam bom sedaj razložil, kako zagnati algoritem, če hočemo napovedati rezultate 
za množico fullTest.csv, ki je množica za backtest.

1. Za meta des potrebujemo 3 množice z atributi, poimenujemo jih XMeta, XSel, XTest
XMeta in XSel sta za potrebe učenja in napovedovanja, XTest pa je množica, katero napovedujemo.
Jaz sem uporabljal za XMeta naključnih 20 000 naključnih primerkov iz druge tretjine množice ostanekTrainFeatures, za XSel sem vzel 20 000 naključnih primerkov iz tretje tretine, za XTest 
pa sem vzel celo FullTest množico(da sem lahko napovedal za backtest).

Vse množice morajo imeti header in so v csv formatu, z imeni XMeta.csv, XSel.csv, XTest.csv v mapi data/dataForMeta/ostanek/
Ime datoteke XTest sem zamenjal z množico FullTestBrezCudni.csv. To specificiramo v metodi wholeMetaProcedureBackTest(), ko kličemo metodo
writeModelPreds() in sicer moramo specificirati parameter dataCsvXTest = folder + "fullTestFeaturesBrezCudni.csv"
folder je trenutno nastavljen na data/dataForMeta/ostanek/

Paziti moramo, da so vse mnozice kompatibilne, torej imajo isto stevilo atributov!

2. Responsi klasifikatorjev morajo biti v mapi data/dataForMeta/ostanek/classifiers/imeKlasifikatorja/
torej mora imeti vsak klasifikator svojo mapo, v kateri morajo biti datoteke: YCaMeta.csv, YCaSel.csv. 
Ti dve datoteki morata imeti response, narejene iz množice XMeta in XSel

3. Responsi klasifikatorjev na množici XTest morajo biti v mapi data/dataForMeta/ostanek/backtest/classifiers/imeKlasifikatorja/
kjer so responsi shranjeni v datoteki z imenom YCaTest.csv. 
Tudi tu morajo responsi biti generirani z modelom na množici XTest (logično)
Ker je ta file lahko kar velik, se responsi generirajo inkrementalno.

4. Zaženi program testingMetaDes.py, kjer je že nastavljeno, da se požene metoda wholeMetaProcedureBackTest(), ki izvede algoritem.

Vbistvu lahko shranjujemo stvari tudi drugače, če v algoritmu spremenimo poti, ampak zaradi lažjega razumevanja upoštevajte te korake, ki sem jih opisal zgoraj

Vse datoteke so za primer pravilno nastavljene na najmočnejšem računalniku z 32gb rama v datotek C:/MartinFreser/ELMEktimo/

V tej metodi lahko spreminjamo parametre metode

	# hc ... consensus tresshold
    # K ... Number of nearest neighbours of Region
    # Kp ... Number of nearest neighbours of Output Region
    # metaCls ... Meta classifier, which decides, if classifiers prediction is competent or not
    # mode ... possible choices are "mean", "majorityVote", "majorityVoteProbs", "weighted", "weightedAll"
            "mean" ... predict mean predictions off all classifiers, who has competence above competenceTresshold
            "majorityVote" ... predict 0 or 1, according to majority votes of competent classifiers
            "majorityVoteProbs" ... predict ration between number of classifiers, who predict above 0.5 and number
                off all classifiers
            "weighted" ... predicts weighted sum of predictions of competent classifiers according to their competence
            "weightedAll" ... same as weighted, except it takes into account all classifiers
        competenceTresshold ... tresshold, whether classifiers are competent or not
        metric ... metric to use to measure distance between examples. Should be compatible with kd_tree or Ball_tree
            metrics.
        metaClsMode ... possible choices:
            "one" ... We use one classifier to compute competence
            "combined" ... We use as many meta classifiers as there are classifiers, so we have one metaClassifier for
                each classifier to tell us, wether classifier is competent or not
