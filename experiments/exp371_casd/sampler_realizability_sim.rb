#!/usr/bin/env ruby
# frozen_string_literal: true

# Reproduce the exp371 audit of how often the repository's camera-agnostic
# RandomIdentitySampler can place exactly three cross-camera donors beside an
# anchor when K=4.  This is a Monte Carlo audit of the grouping rule, not a
# bitwise replay of Python's random.shuffle.

require "json"
require "optparse"

options = {
  epochs: 1000,
  seed_base: 371_000,
  group_size: 4
}

OptionParser.new do |parser|
  parser.banner = "Usage: sampler_realizability_sim.rb [options] [train.list]"
  parser.on("--epochs N", Integer) { |value| options[:epochs] = value }
  parser.on("--seed-base N", Integer) { |value| options[:seed_base] = value }
  parser.on("--group-size N", Integer) { |value| options[:group_size] = value }
end.parse!

abort("group_size must be 4 for the registered exp371 audit") unless options[:group_size] == 4
abort("epochs must be positive") unless options[:epochs].positive?

input_path = ARGV.shift
input = input_path ? File.open(input_path, "r") : STDIN

by_pid = Hash.new { |hash, key| hash[key] = [] }
input.each_line do |line|
  match = line.match(/([-\d]+)_c(\d)/)
  next unless match

  pid = match[1].to_i
  next if pid == -1

  by_pid[pid] << match[2].to_i
end
input.close if input_path

abort("no Duke-format samples found") if by_pid.empty?

def combinations(n, k)
  return 0.0 if n < k || k.negative?
  return 1.0 if k.zero?

  (1..k).inject(1.0) { |value, index| value * (n - k + index) / index }
end

image_count = by_pid.values.sum(&:length)
pid_sizes = by_pid.values.map(&:length)
static_strict3 = 0
closed_form_counts = Hash.new(0.0)
distinct_camera_triplet_probability_sum = 0.0

by_pid.each_value do |cameras|
  n = cameras.length
  camera_counts = Hash.new(0)
  cameras.each { |camera| camera_counts[camera] += 1 }

  cameras.each do |anchor_camera|
    cross_camera_count = n - camera_counts.fetch(anchor_camera)
    static_strict3 += 1 if cross_camera_count >= 3

    denominator = combinations(n - 1, 3)
    (0..3).each do |donor_count|
      numerator = combinations(cross_camera_count, donor_count) *
                  combinations(n - 1 - cross_camera_count, 3 - donor_count)
      closed_form_counts[donor_count] += denominator.positive? ? numerator / denominator : 0.0
    end

    other_camera_counts = camera_counts.reject { |camera, _| camera == anchor_camera }.values
    distinct_numerator = 0.0
    other_camera_counts.combination(3) do |triple|
      distinct_numerator += triple.inject(1.0) { |product, count| product * count }
    end
    distinct_camera_triplet_probability_sum += denominator.positive? ? distinct_numerator / denominator : 0.0
  end
end

aggregate_counts = Hash.new(0)
epoch_coverage = []
used_per_epoch = nil

options[:epochs].times do |epoch|
  rng = Random.new(options[:seed_base] + epoch)
  epoch_counts = Hash.new(0)

  by_pid.each_value do |original_cameras|
    cameras = original_cameras.shuffle(random: rng)
    cameras.each_slice(options[:group_size]) do |group|
      next unless group.length == options[:group_size]

      group.each_with_index do |anchor_camera, anchor_index|
        donor_count = group.each_with_index.count do |donor_camera, donor_index|
          donor_index != anchor_index && donor_camera != anchor_camera
        end
        epoch_counts[:total] += 1
        epoch_counts[donor_count] += 1
      end
    end
  end

  used_per_epoch ||= epoch_counts.fetch(:total)
  abort("used anchor count changed across epochs") unless used_per_epoch == epoch_counts.fetch(:total)

  aggregate_counts[:total] += epoch_counts.fetch(:total)
  (0..3).each { |count| aggregate_counts[count] += epoch_counts[count] }
  epoch_coverage << epoch_counts[3].fdiv(epoch_counts.fetch(:total))
end

strict3_min, strict3_max = epoch_coverage.minmax
strict3_mean = epoch_coverage.sum.fdiv(epoch_coverage.length)
strict3_min_epoch = epoch_coverage.index(strict3_min)
strict3_max_epoch = epoch_coverage.index(strict3_max)

result = {
  "schema_version" => 1,
  "audit" => "exp371_random_identity_sampler_cross_camera_realizability",
  "input" => {
    "path" => input_path,
    "pid_count" => by_pid.length,
    "image_count" => image_count,
    "pid_image_count_min" => pid_sizes.min,
    "pid_image_count_max" => pid_sizes.max
  },
  "registered_protocol" => {
    "epochs" => options[:epochs],
    "epoch_seed_formula" => "seed_base + epoch_index",
    "seed_base" => options[:seed_base],
    "seed_first" => options[:seed_base],
    "seed_last" => options[:seed_base] + options[:epochs] - 1,
    "group_size" => options[:group_size],
    "selected_donors" => 3,
    "camera_rule" => "each donor camera differs from anchor camera",
    "donor_pairwise_camera_distinct_required" => false,
    "rng_note" => "Ruby Random Monte Carlo; not a bitwise replay of Python random.shuffle"
  },
  "simulation" => {
    "used_anchors_per_epoch" => used_per_epoch,
    "tail_dropped_per_epoch" => image_count - used_per_epoch,
    "used_anchors_all_epochs" => aggregate_counts.fetch(:total),
    "donor_count_histogram" => (0..3).to_h { |count| [count.to_s, aggregate_counts[count]] },
    "donor_count_ratio" => (0..3).to_h do |count|
      [count.to_s, aggregate_counts[count].fdiv(aggregate_counts.fetch(:total))]
    end,
    "strict3_epoch_coverage" => {
      "min" => strict3_min,
      "min_epoch_index" => strict3_min_epoch,
      "min_count" => (strict3_min * used_per_epoch).round,
      "mean" => strict3_mean,
      "mean_count" => aggregate_counts[3].fdiv(options[:epochs]),
      "max" => strict3_max,
      "max_epoch_index" => strict3_max_epoch,
      "max_count" => (strict3_max * used_per_epoch).round
    }
  },
  "static_global_pool" => {
    "eligible_anchor_count" => static_strict3,
    "ineligible_anchor_count" => image_count - static_strict3,
    "eligible_anchor_ratio" => static_strict3.fdiv(image_count),
    "definition" => "anchor PID has at least three images from cameras different from the anchor"
  },
  "closed_form_check" => {
    "donor_count_ratio" => (0..3).to_h do |count|
      [count.to_s, closed_form_counts[count].fdiv(image_count)]
    end,
    "strict3_ratio" => closed_form_counts[3].fdiv(image_count),
    "three_donors_from_three_distinct_cameras_ratio" =>
      distinct_camera_triplet_probability_sum.fdiv(image_count)
  }
}

puts JSON.pretty_generate(result)
