package org.mg.javalib.datamining;

import java.awt.Dimension;
import java.io.FileReader;
import java.util.HashMap;
import java.util.LinkedHashMap;

import org.mg.javalib.freechart.FreeChartUtil;
import org.mg.javalib.util.ArrayUtil;
import org.mg.javalib.util.SwingUtil;

import weka.core.Instances;

public class ResultsParser
{
	public static final String[] PERFORMANCE_MEASURES = new String[] { "Accuracy", "AUC", "Sensitivity", "Selectivity",
			"Specificity" };
	//	public static final String[] PERFORMANCE_MEASURES = new String[] { "TP", "FP", "TN", "FN" };

	static String performanceMeasures[];
	static HashMap<String, String> wekaAttributes = new LinkedHashMap<>();
	static
	{
		setPerformanceMeasures(PERFORMANCE_MEASURES);
	}

	public static void setPerformanceMeasures(String perf[])
	{
		performanceMeasures = perf;

		wekaAttributes.clear();
		wekaAttributes.put("Key_Dataset", "Dataset");
		wekaAttributes.put("Key_Run", "Run");
		wekaAttributes.put("Key_Fold", "Fold");
		wekaAttributes.put("measureNumAttributesSelected", "Features");

		if (ArrayUtil.indexOf(perf, "AUC") != -1)
			wekaAttributes.put("Area_under_ROC", "AUC");
		if (ArrayUtil.indexOf(perf, "Accuracy") != -1)
			wekaAttributes.put("Percent_correct", "Accuracy");

		if (ArrayUtil.indexOf(perf, "Sensitivity") != -1)
			wekaAttributes.put("True_positive_rate", "Sensitivity");

		if (ArrayUtil.indexOf(perf, "Selectivity") != -1)
			wekaAttributes.put("IR_precision", "Selectivity");

		if (ArrayUtil.indexOf(perf, "Specificity") != -1)
			wekaAttributes.put("True_negative_rate", "Specificity");

		if (ArrayUtil.indexOf(perf, "FMeasure") != -1)
			wekaAttributes.put("F_measure", "FMeasure");
		if (ArrayUtil.indexOf(perf, "TP") != -1)
			wekaAttributes.put("Num_true_positives", "TP");
		if (ArrayUtil.indexOf(perf, "TN") != -1)
			wekaAttributes.put("Num_true_negatives", "TN");
		if (ArrayUtil.indexOf(perf, "FP") != -1)
			wekaAttributes.put("Num_false_positives", "FP");
		if (ArrayUtil.indexOf(perf, "FN") != -1)
			wekaAttributes.put("Num_false_negatives", "FN");
		if (ArrayUtil.indexOf(perf, "AUP") != -1)
			wekaAttributes.put("Area_under_PRC", "AUP");

		wekaAttributes.put("Correlation_coefficient", "Pearson");
		wekaAttributes.put("Spearman_correlation", "Spearman");
		wekaAttributes.put("Root_mean_squared_error", "RMSE");

		wekaAttributes.put("Key_Scheme", "Algorithm");
	}

	public static void parse(String arffResultFiles[], String baseName, String props[]) throws Exception
	{
		HashMap<String, String[]> p = new HashMap<>();
		p.put("Props", props);
		parse(arffResultFiles, baseName, p);
	}

	public static void parse(String arffResultFiles[], String baseName, HashMap<String, String[]> props)
			throws Exception
	{
		//		File arff;
		//		if (arffResultFiles.length > 1)
		//		{
		//			arff = File.createTempFile("result", "arff");
		//			arff.deleteOnExit();
		//			MergeArffFiles.merge(arffResultFiles, arff.getAbsolutePath());
		//		}
		//		else
		//			arff = new File(arffResultFiles[0]);

		ResultSet allResults = null;
		for (String arff : arffResultFiles)
		{
			System.out.println("reading " + arff);
			if (allResults == null)
				allResults = WekaResultSetUtil.fromWekaDataset(new Instances(new FileReader(arff)));
			else
				allResults.concat(WekaResultSetUtil.fromWekaDataset(new Instances(new FileReader(arff))));
		}

		for (String name : props.keySet())
		{
			setPerformanceMeasures(props.get(name));

			ResultSet results = new ResultSet();
			for (int i = 0; i < allResults.getNumResults(); i++)
			{
				int idx = results.addResult();
				for (String wp : wekaAttributes.keySet())
					results.setResultValue(idx, wekaAttributes.get(wp), allResults.getResultValue(i, wp));
			}
			if (wekaAttributes.containsKey("Percent_correct") && results.getProperties().contains("Percent_correct"))
				for (int i = 0; i < results.getNumResults(); i++)
					results.setResultValue(i, wekaAttributes.get("Percent_correct"),
							((Double) results.getResultValue(i, wekaAttributes.get("Percent_correct"))) * 0.01);

			if (wekaAttributes.containsKey("Key_Scheme"))
				for (int i = 0; i < results.getNumResults(); i++)
				{
					String s = results.getResultValue(i, wekaAttributes.get("Key_Scheme")).toString();
					s = s.substring(s.lastIndexOf('.') + 1);
					results.setResultValue(i, wekaAttributes.get("Key_Scheme"), s);
				}

			System.out.println(results.toNiceString());

			//			ResultSetBoxPlot plot = new ResultSetBoxPlot(results, "", "", "Dataset",
			//					ArrayUtil.toList(performanceMeasures));
			ResultSetBoxPlot plot = new ResultSetBoxPlot(results, "", "", "Dataset", "Algorithm",
					performanceMeasures[0]);

			plot.setHideMean(true);
			plot.printNumResultsPerPlot(true);
			plot.setPrintMeanAndStdev(true);
			//			plot.setYRange(0.81, 0.82);
			FreeChartUtil.toPNGFile(baseName + ".png", plot.getChart(), new Dimension(800, 600));
			SwingUtil.showInFrame(plot.getChart(), new Dimension(800, 600));
		}
	}

	public static void main(String[] args) throws Exception
	{
		//ResultsParser.parse("/tmp/outfile1", "/tmp/outfile2");
	}

}
