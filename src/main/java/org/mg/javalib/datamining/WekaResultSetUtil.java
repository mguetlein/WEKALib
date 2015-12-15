package org.mg.javalib.datamining;

import org.mg.javalib.datamining.ResultSet.TTester;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.experiment.PairedStats;
import weka.experiment.PairedStatsCorrected;

public class WekaResultSetUtil
{
	public static ResultSet fromWekaDataset(Instances data)
	{
		ResultSet set = new ResultSet();
		for (int a = 0; a < data.numAttributes(); a++)
			set.properties.add(data.attribute(a).name());

		for (Instance instance : data)
		{
			Result r = new Result();
			for (int a = 0; a < data.numAttributes(); a++)
			{
				Attribute att = data.attribute(a);
				if (att.isNumeric())
					r.setValue(att.name(), instance.value(att));
				else
					r.setValue(att.name(), instance.stringValue(att));
			}
			set.results.add(r);
		}
		return set;
	}
	
	public static TTester T_TESTER = new TTester(){
		public int ttest(double v1[], double v2[], double mean1, double mean2, double confidence, Double correctTerm)
		{
			PairedStats pairedStats;
			if (correctTerm == null)
				pairedStats = new PairedStats(confidence);
			else
				pairedStats = new PairedStatsCorrected(confidence, correctTerm);
			pairedStats.add(v1, v2);
			pairedStats.calculateDerived();
			int test = 0;
			if (pairedStats.differencesSignificance > 0)
				test = 1;
			else if (pairedStats.differencesSignificance < 0)
				test = -1;
			return test;
		}
	};

}
