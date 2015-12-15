package org.mg.wekalib.classifier;

import java.io.FileReader;
import java.util.ArrayList;
import java.util.List;

import org.mg.javalib.util.ListUtil;
import org.mg.wekalib.data.InstanceUtil;

import weka.attributeSelection.PrincipalComponents;
import weka.classifiers.ConditionalDensityEstimator;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class PCAClassifier extends TransformDataClassifier implements ConditionalDensityEstimator
{
	List<Double> explainedVariance;
	List<String> attributeNames;

	List<PrincipalComponents> pc = new ArrayList<>();
	List<Integer> numPCs = new ArrayList<>();

	public PCAClassifier(List<Double> explainedVariance, List<String> attributeNames)
	{
		this.explainedVariance = explainedVariance;
		this.attributeNames = attributeNames;
	}

	@Override
	public Instances transformData(Instances data, boolean train) throws Exception
	{
		if (train)
			System.out.println("num attributes orig w/o class: " + (data.numAttributes() - 1));

		List<Instances> blocks = new ArrayList<>();

		int attrIdx = 0;

		for (int i = 0; i < attributeNames.size(); i++)
		{
			String name = attributeNames.get(i);
			Double explainedVar = explainedVariance.get(i);

			List<Attribute> exAttributes = new ArrayList<>();
			List<Attribute> pcaAttributes = new ArrayList<>();
			for (; attrIdx < data.numAttributes(); attrIdx++)
			{
				if (!data.attribute(attrIdx).name().matches(name))
				{
					if (pcaAttributes.size() > 0)
						break;
					exAttributes.add(data.attribute(attrIdx));
				}
				else
					pcaAttributes.add(data.attribute(attrIdx));
			}

			if (exAttributes.size() > 0)
			{
				if (train)
					System.out.println("add " + exAttributes.size() + " un-transformed attributes");
				blocks.add(InstanceUtil.getAttributes(data, exAttributes));
			}

			if (pcaAttributes.size() > 0)
			{
				Instances pcaIn = InstanceUtil.getAttributes(data, pcaAttributes);
				if (train)
					System.out.println("num attributes pca-in: " + pcaIn.numAttributes());

				PrincipalComponents pca = new PrincipalComponents();
				if (train)
				{
					pc.add(pca);
					pca.setVarianceCovered(explainedVar);
					pca.buildEvaluator(pcaIn);
				}
				else
				{
					pca = pc.get(i);
				}
				Instances pcaAll = pca.transformedData(pcaIn);

				int numPCs;
				if (train)
				{
					numPCs = pcaAll.numAttributes();
					for (int a = 0; a < pcaAll.numAttributes(); a++)
					{
						if (pcaAll.variance(a) == 0)
						{
							numPCs = a;
							break;
						}
					}
					this.numPCs.add(numPCs);
				}
				else
				{
					numPCs = this.numPCs.get(i);
				}

				Instances pcaSelected;
				if (pcaAll.numAttributes() > numPCs)
				{
					Remove remove = new Remove();
					remove.setAttributeIndices((numPCs + 1) + "-last"); // retain first X PCs!
					remove.setInputFormat(pcaAll);
					pcaSelected = Filter.useFilter(pcaAll, remove);
				}
				else
					pcaSelected = pcaAll;
				if (train)
					System.out.println("num attributes pca-out: " + pcaSelected.numAttributes());
				blocks.add(pcaSelected);
			}
		}
		if (attrIdx < data.numAttributes())
		{
			List<Attribute> exAttributes = new ArrayList<>();
			for (; attrIdx < data.numAttributes(); attrIdx++)
				exAttributes.add(data.attribute(attrIdx));
			if (train)
				System.out.println("add " + exAttributes.size() + " un-transformed attributes");
			blocks.add(InstanceUtil.getAttributes(data, exAttributes));
		}

		Instances transformed = null;
		for (Instances d : blocks)
		{
			if (transformed == null)
				transformed = d;
			else
				InstanceUtil.concatColumns(transformed, d);
		}

		transformed.setClassIndex(transformed.numAttributes() - 1);

		if (train)
			System.out.println("num attributes transformed: " + transformed.numAttributes());

		//		System.out.println(transformed);

		return transformed;
	}

	@Override
	public double logDensity(Instance instance, double value) throws Exception
	{
		if (!(m_Classifier instanceof ConditionalDensityEstimator))
			return Double.NaN;
		else
			return ((ConditionalDensityEstimator) m_Classifier).logDensity(instance, value);
	}

	public static void main(String[] args) throws Exception
	{
		Instances inst = new Instances(new FileReader(System.getProperty("user.home")
				+ "/data/weka/numeric/auto93.arff"));
		System.out.println("instances: " + inst.numInstances());
		System.out.println("attributes: " + inst.numInstances());

		List<Double> d = ListUtil.createList((Double) 0.5, (Double) 0.2);
		List<String> a = new ArrayList<>();
		a.add("City_MPG|Highway_MPG|Air_Bags_standard");
		a.add("Length|Wheelbase|Width");
		PCAClassifier c = new PCAClassifier(d, a);
		c.buildClassifier(inst);

	}
}