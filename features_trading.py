#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2022 Baidu.com, Inc. All Rights Reserved
#
########################################################################
"""
File: features_tradings.py
Author: work(work@baidu.com)
Date: 2022/06/08 19:13:32
"""
import os
import sys
import jieba
import jieba.analyse
from collections import defaultdict

try:
    from domain_hot_event.util.similarity_auxiliary import SimilarityAuxiliary
except ImportError as e:
    raise ImportError("Unable to import related module which is need {}".format(e))


class FeaturesTrading(object):
    """
    特征落盘
    """
    def __init__(self, cluster_obj):
        """
        Args:
            cluster_obj: 事件或角度对象
        """
        self._features = dict()
        self._cluster_obj = cluster_obj
        self._nid_assemble = list()
        self._comprehensive_bangdan = {"toutiao": "头条", "tencent": "腾讯", "weibo": "微博", \
                            "douyin": "抖音", "kuaishou": "快手"}

    def __str__(self):
        """__str__"""
        return 'topic or event: {}'.format(self._cluster_obj.eid)
    
    def _get_eid(self):
        """获取eid"""
        self._features['eid'] = self._cluster_obj.eid

    def _get_hot_list(self):
        """获取榜单信号"""
        self._features['bangdan'] = list()
        bangdan = self._cluster_obj.sim_hot_list
        for k, v in self._cluster_obj.sim_domain_hot_list.items():
            bangdan = bangdan | v
        
        for hot_list_obj in bangdan:
            _domain = hot_list_obj.domain.value.encode('utf-8')
            _source = self._comprehensive_bangdan[hot_list_obj.source]
            _title = hot_list_obj.title.encode('utf-8')
            _rank = hot_list_obj.rank

            self._features['bangdan'].append('--'.join(map(str, [_source, _domain, _rank, _title])))

    def _get_query(self):
        """获取query信号"""
        self._features['sim_query'] = list()
        for query_obj in self._cluster_obj.sim_querys:
            self._features['sim_query'].append(query_obj.title.encode('utf-8'))

    def _get_inter_level_1(self):
        """获取一级干预信号"""
        self._features['intervention_level_1'] = self._cluster_obj.intervention_level_1

    def _get_inter_level_2(self):
        """获取二级干预信号"""
        self._features['intervention_level_2'] = self._cluster_obj.intervention_level_2
    
    def _get_imp_bjh_num(self):
        """获取重要百家号个数"""
        self._features['imp_bjh_num'] = self._cluster_obj.important_bjh_num
    

class TopicFeaturesTrading(FeaturesTrading):
    """
    事件相关特征落盘
    """
    def __init__(self, cluster_obj):
        """__init__"""
        super(TopicFeaturesTrading, self).__init__(cluster_obj)

    def __str__(self):
        """__str__"""
        return 'topic: {}'.format(self._cluster_obj.eid)

    def __call__(self):
        """__call__"""
        return self.__load()

    def _get_topic_event_time_date(self):
        """获取事件开始时间"""
        self._features['event_time_date'] = self._cluster_obj.event_time_date

    def _get_topic_cluster_size(self):
        """获取事件size"""
        _event_cluster = self._cluster_obj.event_cluster_list
        for event_cluster_obj in _event_cluster:
            self._nid_assemble.extend(list(event_cluster_obj.items))

        self._features['topic_cluster_size'] = len(self._nid_assemble)

    def _get_topic_domain(self):
        """获取事件领域"""
        _event_domain = list()
        for domain in self._cluster_obj.event_domain:
            _event_domain.append(domain.encode('utf-8'))

        _event_domain_str = '--'.join(_event_domain)
        self._features['event_domain'] = _event_domain_str

    def _get_topic_domain_level(self):
        """获取事件等级"""
        self._features['domain_level'] = defaultdict(str)
        for domain, level in self._cluster_obj.domain_level.items():
            assert level in ['9', '99', '999']
            self._features['domain_level'][domain.value.encode('utf-8')] = level

    def _get_topic_final_score(self):
        """获取final_score"""
        self._features['final_score'] = defaultdict(float)
        for cate, score in self._cluster_obj.final_score.items():
            self._features['final_score'][cate.encode('utf-8')] = score

    def _get_topic_title(self):
        """获取事件title"""
        _title_list = list()
        for nid_obj in self._nid_assemble:
            _title_list.extend(nid_obj.seg_words)

        _title_str = ','.join(map(lambda x:x.encode('utf-8'), _title_list))

        res_tfidf = jieba.analyse.extract_tags(_title_str, topK=5, withWeight=False)
        res_tfidf_str = '--'.join(map(lambda x:x.encode('utf-8'), res_tfidf))
        self._features['title_seg_words'] = res_tfidf_str

        ## 提取出5个和tfidf相似的nid标题
        _nid_tfidf_sim = defaultdict(float)
        for nid_obj in self._nid_assemble:
            _title = nid_obj.title.encode('utf-8')
            _nid_title = '--'.join(map(str, [nid_obj.nid, _title]))
            nid_obj_seg_words = nid_obj.seg_words
            sim_seg_words = SimilarityAuxiliary.jarcard_sim(nid_obj_seg_words, res_tfidf)
            _nid_tfidf_sim[_nid_title] = sim_seg_words

        sorted(_nid_tfidf_sim.items(), key = lambda x: x[1], reverse = True)
        self._features['title_list_5_quota'] = _nid_tfidf_sim.keys()[:5]

    def __load(self):
        self._get_eid()
        self._get_topic_event_time_date()
        self._get_topic_cluster_size()
        self._get_hot_list()
        self._get_query()
        self._get_inter_level_1()
        self._get_inter_level_2()
        self._get_imp_bjh_num()
        self._get_topic_domain()
        self._get_topic_domain_level()
        self._get_topic_final_score()
        self._get_topic_title()
        
        return self._features
        

class EventFeaturesTrading(FeaturesTrading):
    """
    角度相关特征落盘
    """
    pass
