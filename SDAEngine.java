import org.canova.api.conf.Configuration;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.split.FileSplit;
import org.canova.nd4j.nlp.reader.TfidfRecordReader;
import org.canova.nd4j.nlp.vectorizer.TfidfVectorizer;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.AutoEncoder;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


public class SDAEngine {

    private static Logger log = LoggerFactory.getLogger(SDAEngine.class);


    public static void main(String[] args) throws Exception {

        int outputNum = 2;


        File trainDirectory = new File("/Users/akshitatyagi/Downloads/Corpuses/Train");
        File testDirectory = new File("/Users/akshitatyagi/Downloads/Corpuses/Train");

        int batchSize = 50;
        List<String> labels = Arrays.asList(trainDirectory.list());
        int numLabels = labels.size();


        log.info("Loading the train data....");
        Configuration config = new Configuration();
        config.setInt(TfidfVectorizer.MIN_WORD_FREQUENCY, 15);
        config.setBoolean(RecordReader.APPEND_LABEL, true);

        TfidfRecordReader trainReader = new TfidfRecordReader();
        trainReader.initialize(config, new FileSplit(trainDirectory));                                  // Labeled path to the training vectors
        DataSetIterator trainIter = new RecordReaderDataSetIterator(trainReader, batchSize, -1, numLabels);


        int inputNum = trainReader.getNumFeatures();
        log.info("Number of Features: " + inputNum);


        log.info("Loading the test data...");
        TfidfRecordReader testReader = new TfidfRecordReader();
        testReader.setTfidfVectorizer(trainReader.getTfidfVectorizer());                                //reuse vectorizer
        testReader.initialize(config, new FileSplit(testDirectory));                                    // Labeled path to the text vectors
        DataSetIterator testIter = new RecordReaderDataSetIterator(testReader, batchSize, -1, numLabels);


        log.info("Build model....");

        int iterations = 1;
        int seed = 123;
        WeightInit weightInit = WeightInit.XAVIER;
        String activation = "relu";
        Updater updater = Updater.NESTEROVS;
        double lr = 1e-4;                                                                          
        double mu = 0.9;                                                                            
        double l2 = 5e-6;                                                                          
        boolean regularization = true;

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(iterations)
                .activation(activation)
                .updater(updater)
                .gradientNormalization(GradientNormalization.RenormalizeL2PerLayer)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .gradientNormalizationThreshold(1.0)
                .learningRate(lr)
                .momentum(mu)
                .regularization(regularization)
                .l2(l2)

                .list()
                .layer(0, new AutoEncoder.Builder()
                        .nIn(inputNum)
                        .nOut(5000)
                        .weightInit(weightInit)
                        .lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build()
                )
                .layer(1, new AutoEncoder.Builder()
                        .nIn(5000)
                        .nOut(2000)
                        .weightInit(weightInit)
                        .lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build()
                )
                .layer(2, new AutoEncoder.Builder()
                        .nIn(2000)
                        .nOut(1000)
                        .weightInit(weightInit)
                        .lossFunction(LossFunction.RMSE_XENT)
                        .corruptionLevel(0.3)
                        .build()
                )
                .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                        .activation("softmax")
                        .nIn(1000)
                        .nOut(outputNum)
                        .build()
                )
                .pretrain(true)
                .backprop(false)
                .build();


        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(1)));
        //model.setListeners(new HistogramIterationListener(1));                                                // Generating the Visualizing model


        log.info("Train model....");
        int epochs = 5;
        for (int i = 0; i < epochs; i++) {
            trainIter.reset();
            model.fit(trainIter);
            log.info("*** Completed epoch {} ***", i);
        }

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);
        while (testIter.hasNext()) {
            DataSet next = testIter.next();
            INDArray features = next.getFeatureMatrix();
            INDArray Labels = next.getLabels();
            INDArray prediction = model.output(features, false);
            eval.eval(Labels, prediction);
        }
        log.info(eval.stats());
        log.info("**************** FINISH ********************");

    }
}
