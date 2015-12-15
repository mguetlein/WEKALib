package org.mg.wekalib.evaluation.alt;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import org.mg.javalib.util.ListUtil;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.evaluation.NominalPrediction;
import weka.core.Instances;
import weka.core.Utils;

public class ExtendedEvaluation extends Evaluation
{
	Instances data;

	public ExtendedEvaluation(Instances data) throws Exception
	{
		super(data);
		this.data = data;
	}

	public double enrichmentFactor(final int classIndex, double percent)
	{
		if (m_Predictions == null)
		{
			return Utils.missingValue();
		}
		else
		{
			int allTotal = 0;
			int allClass = 0;
			int erTotal = 0;
			int erClass = 0;
			List<NominalPrediction> l = ListUtil.cast(NominalPrediction.class, m_Predictions);
			Collections.sort(l, new Comparator<NominalPrediction>()
			{
				@Override
				public int compare(NominalPrediction o1, NominalPrediction o2)
				{
					return Double.compare(o2.distribution()[classIndex], o1.distribution()[classIndex]);
				}
			});
			for (NominalPrediction p : l)
			{
				if (allTotal < Math.round(l.size() * percent))
				{
					if (p.actual() == classIndex)
						erClass++;
					erTotal++;
				}
				if (p.actual() == classIndex)
					allClass++;
				allTotal++;
			}
			double allRatio = allClass / (double) allTotal;
			double erRatio = erClass / (double) erTotal;
			return erRatio / allRatio;
		}
	}

}
