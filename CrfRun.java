package com.suning.spark.crf;

import com.intel.ssg.bdt.nlp.CRF;
import com.intel.ssg.bdt.nlp.CRFModel;
import com.intel.ssg.bdt.nlp.Sequence;
import org.apache.spark.SparkConf;
import org.apache.spark.SparkContext;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.Function2;
import org.apache.spark.api.java.function.PairFunction;
import org.apache.spark.api.java.function.VoidFunction;
import scala.Tuple2;

import java.util.Arrays;

/**
 * @Author ZhangJiawei
 * @Date 2017/1/9.
 */
public class CrfRun {
    public static void main(String[] args) {
        SparkConf sc = new SparkConf();
        sc.setMaster("local");
        sc.setAppName("CrfRun");
        SparkContext sparkContext = new SparkContext(sc);
        JavaSparkContext javaSparkContext = JavaSparkContext.fromSparkContext(sparkContext);
        //模板
        String[] template = {"U00:%x[-1,0]", "U01:%x[0,0]", "U02:%x[1,0]", "B"};
//训练样本，语义分析样本格式说明（tag|--|tag|-|term）
        String[] train = {"pmb|--|pmb|-|沃玛\tnone|--|none|-|虎\tnone|--|none|-|鲨\tnone|--|none|-|号", "pmc|--|pmc|-|衬衫\tnone|--|none|-|短袖\tnone|--|none|-|男"};
//测试样本
        String[] test = {"pmc|--|null|-|沃玛\tpmb|--|null|-|衬衫"};
//训练样本数据转化
        JavaRDD<Sequence> trainRdd = javaSparkContext
                .parallelize(Arrays.asList(train))
                .map(new Function<String, Sequence>() {
                    public Sequence call(String v1) throws Exception {
                        return Sequence.deSerializer(v1);
                    }
                });
//测试样本数据转化
        JavaRDD<Sequence> testRdd = javaSparkContext
                .parallelize(Arrays.asList(test))
                .map(new Function<String, Sequence>() {
                    public Sequence call(String v1) throws Exception {
                        return Sequence.deSerializer(v1);
                    }
                });
//模型训练
        CRFModel crfModel = CRF.train(template, trainRdd.rdd());
//输出结果
        JavaRDD<Sequence> result = crfModel.predict(testRdd.rdd()).toJavaRDD();

        result.foreach(new VoidFunction<Sequence>() {
            public void call(Sequence sequence) throws Exception {
                System.out.println(sequence);
            }
        });
//转化操作
        JavaPairRDD<Long, Sequence> resultSwapRDD = result.zipWithIndex().mapToPair(new PairFunction<Tuple2<Sequence, Long>, Long, Sequence>() {
            public Tuple2<Long, Sequence> call(Tuple2<Sequence, Long> sequenceLongTuple2) throws Exception {
                return new Tuple2<Long, Sequence>(sequenceLongTuple2._2(), sequenceLongTuple2._1());
            }
        });

//转化操作
        JavaPairRDD<Long, Sequence> testSwapRDD = testRdd.zipWithIndex().mapToPair(new PairFunction<Tuple2<Sequence, Long>, Long, Sequence>() {
            public Tuple2<Long, Sequence> call(Tuple2<Sequence, Long> sequenceLongTuple2) throws Exception {
                return new Tuple2<Long, Sequence>(sequenceLongTuple2._2(), sequenceLongTuple2._1());
            }
        });

//计算标记正确的数目
        Integer score = resultSwapRDD.join(testSwapRDD).map(new Function<Tuple2<Long, Tuple2<Sequence, Sequence>>, Tuple2<Sequence, Sequence>>() {
            public Tuple2<Sequence, Sequence> call(Tuple2<Long, Tuple2<Sequence, Sequence>> v1) throws Exception {
                return v1._2();
            }
        }).map(new Function<Tuple2<Sequence, Sequence>, Integer>() {
            public Integer call(Tuple2<Sequence, Sequence> v1) throws Exception {
                return v1._1().compare(v1._2());
            }
        }).reduce(new Function2<Integer, Integer, Integer>() {
            public Integer call(Integer v1, Integer v2) throws Exception {
                return v1 + v2;
            }
        });

//计算总数目
        Integer total = testRdd.map(new Function<Sequence, Integer>() {
            public Integer call(Sequence v1) throws Exception {
                return v1.toArray().length;
            }
        }).reduce(new Function2<Integer, Integer, Integer>() {
            public Integer call(Integer v1, Integer v2) throws Exception {
                return v1 + v2;
            }
        });

        System.out.println("Prediction Accuracy:" + score * 1.0 / total);
    }
}
