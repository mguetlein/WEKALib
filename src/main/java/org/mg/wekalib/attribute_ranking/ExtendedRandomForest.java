package org.mg.wekalib.attribute_ranking;

import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.mg.javalib.datamining.ResultSet;

import weka.classifiers.meta.Bagging;
import weka.classifiers.trees.RandomForest;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

public class ExtendedRandomForest extends RandomForest implements AttributeProvidingClassifier
{
	private static final long serialVersionUID = 3L;

	@Override
	public Set<Integer> getAttributesEmployedForPrediction(Instance instance)
	{
		//		HashMap<Integer, List<Integer>> attDepths = new HashMap<Integer, List<Integer>>();
		Set<Integer> l = new HashSet<Integer>();
		for (int i = 0; i < m_bagger.getNumIterations(); i++)
		{
			ExtendedRandomTree t = ((TreeBagging) m_bagger).getTree(i);
			List<Integer> atts = t.getUsedAttribute(instance);
			//			System.out.println(t);
			//			System.out.println(ListUtil.toString(atts));
			//			for (int j = 0; j < atts.size(); j++)
			//			{
			//				if (!attDepths.containsKey(atts.get(j)))
			//					attDepths.put(atts.get(j), new ArrayList<Integer>());
			//				attDepths.get(atts.get(j)).add(j);
			//			}
			l.addAll(atts);
			//			System.out.println(HashMapUtil.toString(attDepths));
		}
		return l;
	}

	public void printTrees()
	{
		for (int i = 0; i < m_bagger.getNumIterations(); i++)
		{
			ExtendedRandomTree t = ((TreeBagging) m_bagger).getTree(i);
			System.out.println(t);
		}
	}

	class TreeBagging extends Bagging
	{
		public ExtendedRandomTree getTree(int index)
		{
			return (ExtendedRandomTree) m_Classifiers[index];
		}
	}

	public void buildClassifier(Instances data) throws Exception
	{

		// can classifier handle the data?
		getCapabilities().testWithFail(data);

		// remove instances with missing class
		data = new Instances(data);
		data.deleteWithMissingClass();

		m_bagger = new TreeBagging();

		// RandomTree implements WeightedInstancesHandler, so we can
		// represent copies using weights to achieve speed-up.
		m_bagger.setRepresentCopiesUsingWeights(true);

		ExtendedRandomTree rTree = new ExtendedRandomTree();

		// set up the random tree options
		m_KValue = m_numFeatures;
		if (m_KValue < 1)
		{
			m_KValue = (int) Utils.log2(data.numAttributes()) + 1;
		}
		rTree.setKValue(m_KValue);
		rTree.setMaxDepth(getMaxDepth());
		rTree.setDoNotCheckCapabilities(true);

		// set up the bagger and build the forest
		m_bagger.setClassifier(rTree);
		m_bagger.setSeed(m_randomSeed);
		m_bagger.setNumIterations(m_numTrees);
		m_bagger.setCalcOutOfBag(true);
		m_bagger.setNumExecutionSlots(m_numExecutionSlots);
		m_bagger.buildClassifier(data);
	}

	public ResultSet getSummary(boolean nice)
	{
		ResultSet set = new ResultSet();
		int idx = set.addResult();
		set.setResultValue(idx, "Classifier", getName());
		if (!nice)
			set.setResultValue(idx, "Num trees", m_numTrees);
		return set;
	}

	public String getName()
	{
		return "Random Forest";
	}

}
