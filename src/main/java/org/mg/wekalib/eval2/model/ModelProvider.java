package org.mg.wekalib.eval2.model;

import java.util.ArrayList;
import java.util.List;

import org.mg.javalib.util.ArrayUtil;

import weka.classifiers.functions.supportVector.PolyKernel;
import weka.classifiers.functions.supportVector.RBFKernel;

public class ModelProvider
{
	public static final NaiveBayesModel NAIVE_BAYES = new NaiveBayesModel();
	public static final RandomForestModel RANDOM_FOREST = new RandomForestModel();
	public static final SupportVectorMachineModel SVM = new SupportVectorMachineModel();
	public static final SupportVectorMachineModel[] SVMS_PARAM_OPTIMIZE;
	public static final SupportVectorMachineModel[] SVMS_RIDGE_EVAL;
	public static final Model[] ALL_MODELS_PARAM_OPTIMIZE;
	public static final Model[] ALL_MODELS_PARAM_DEFAULT;

	static
	{
		List<SupportVectorMachineModel> svms = new ArrayList<SupportVectorMachineModel>();
		Double cs[] = new Double[] { 1.0, 10.0, 100.0 };
		for (Double g : new Double[] { 0.001, 0.01, 0.1 })
		{
			for (Double c : cs)
			{
				if (c == 1.0 && g == 0.001) // does not work well
					continue;
				SupportVectorMachineModel svm = new SupportVectorMachineModel();
				svm.setC(c);
				svm.setKernel(new RBFKernel());
				svm.setGamma(g);
				svms.add(svm);
			}
		}
		for (Double e : new Double[] { 1.0 }) // exponent optimizing not needed , 2.0, 3.0
		{
			for (Double c : cs)
			{
				SupportVectorMachineModel svm = new SupportVectorMachineModel();
				svm.setC(c);
				svm.setKernel(new PolyKernel());
				svm.setExp(e);
				svms.add(svm);
			}
		}
		SVMS_PARAM_OPTIMIZE = ArrayUtil.toArray(svms);

		svms = new ArrayList<SupportVectorMachineModel>();
		//for (Double r : new Double[] { 1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1 })
		for (Double r : new Double[] { 1e-8, 1e-1 })
		{
			SupportVectorMachineModel svm = new SupportVectorMachineModel();
			svm.setRidge(r);
			svm.setKernel(new RBFKernel());
			svms.add(svm);
		}
		SVMS_RIDGE_EVAL = ArrayUtil.toArray(svms);

		ALL_MODELS_PARAM_OPTIMIZE = ArrayUtil.concat(Model.class,
				new Model[] { NAIVE_BAYES, RANDOM_FOREST }, SVMS_PARAM_OPTIMIZE);
		ALL_MODELS_PARAM_DEFAULT = new Model[] { NAIVE_BAYES, RANDOM_FOREST, SVM };
	}
}
