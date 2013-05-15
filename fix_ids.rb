#!/usr/bin/env ruby

require 'mongo'
require 'ruby-progressbar'

conn = Mongo::Connection.new
db   = conn['yelp']
coll = db['reviews']

user_ids, biz_ids = {}, {}

progress = ProgressBar.create(title: "Reviews", starting_at: 0, total: coll.count)

coll.find.each do |review|
  user_id = (user_ids[review['user_id']] ||= user_ids.count)
  biz_id  = (biz_ids[review['business_id']] ||= biz_ids.count)
  coll.update({'_id' => review['_id']}, {'$set' => {'user_id' => user_id, 'business_id' => biz_id}})
  progress.increment
end
