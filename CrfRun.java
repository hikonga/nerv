package com.suning.spark.crf;

import com.suning.ssg.bdt.nlp.CRF;
import com.suning.ssg.bdt.nlp.CRFModel;
import com.suning.ssg.bdt.nlp.Sequence;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.*;
import scala.Tuple2;

import java.util.Arrays;

/**
 * Created by 14074626 on 2016/12/27.
 */
public class CRFTest {

    public static void main(String []args){

        SparkConf sparkConf = new SparkConf().setAppName("CRFTest").setMaster("local[2]");

        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);

        String [] template = {"U00:%x[-1,0]", "U01:%x[0,0]", "U02:%x[1,0]", "B"};
        String [] train = {"pmb|--|pmb|-|沃玛\tnone|--|none|-|虎\tnone|--|none|-|鲨\tnone|--|none|-|号"
                ,"pmc|--|pmc|-|衬衫\tnone|--|none|-|短袖\tnone|--|none|-|男"};
        String [] test = {"pmb|--|null|-|沃玛\tpmc|--|null|-|衬衫\tnone|--|null|-|男"};

        JavaRDD<Sequence> trainRdd = javaSparkContext
                .parallelize(Arrays.asList(train))
                .map(new Function<String, Sequence>() {
                    public Sequence call(String v1) throws Exception {
                        return Sequence.deSerializer(v1);
                    }
                });

        JavaRDD<Sequence> testRdd = javaSparkContext
                .parallelize(Arrays.asList(test))
                .map(new Function<String, Sequence>() {
                    public Sequence call(String v1) throws Exception {
                        return Sequence.deSerializer(v1);
                    }
                });

        CRFModel crfModel = CRF.train(template,trainRdd);

        JavaRDD<Sequence> result = crfModel.predict(testRdd);

        //转化操作
        JavaPairRDD<Long,Sequence> resultSwapRDD = result.zipWithIndex().mapToPair(new PairFunction<Tuple2<Sequence,Long>, Long, Sequence>() {
            public Tuple2<Long, Sequence> call(Tuple2<Sequence, Long> sequenceLongTuple2) throws Exception {
                return new Tuple2<Long, Sequence>(sequenceLongTuple2._2(),sequenceLongTuple2._1());
            }
        });

        //转化操作
        JavaPairRDD<Long,Sequence> testSwapRDD = testRdd.zipWithIndex().mapToPair(new PairFunction<Tuple2<Sequence,Long>, Long, Sequence>() {
            public Tuple2<Long, Sequence> call(Tuple2<Sequence, Long> sequenceLongTuple2) throws Exception {
                return new Tuple2<Long, Sequence>(sequenceLongTuple2._2(),sequenceLongTuple2._1());
            }
        });

        //计算标记正确的数目
        Integer score = resultSwapRDD.join(testSwapRDD).map(new Function<Tuple2<Long,Tuple2<Sequence,Sequence>>, Tuple2<Sequence,Sequence>>() {
            public Tuple2<Sequence, Sequence> call(Tuple2<Long, Tuple2<Sequence, Sequence>> v1) throws Exception {
                return v1._2();
            }
        }).map(new Function<Tuple2<Sequence,Sequence>, Integer>() {
            public Integer call(Tuple2<Sequence, Sequence> v1) throws Exception {
                return v1._1().compare(v1._2());
            }
        }).reduce(new Function2<Integer, Integer, Integer>() {
            public Integer call(Integer v1, Integer v2) throws Exception {
                return v1+v2;
            }
        });

        //计算总数目
        Integer total = testRdd.map(new Function<Sequence, Integer>() {
            public Integer call(Sequence v1) throws Exception {
                return v1.toArray().length;
            }
        }).reduce(new Function2<Integer, Integer, Integer>() {
            public Integer call(Integer v1, Integer v2) throws Exception {
                return v1+v2;
            }
        });

        System.out.println("Prediction Accuracy:" + score * 1.0/total);

        javaSparkContext.close();

    }
}
