package org.mg.wekalib.classifier;

import java.io.Serializable;

import weka.core.Instance;

public interface LocalModelClusterer extends Serializable
{
	public int clusterIdx(Instance inst);
}