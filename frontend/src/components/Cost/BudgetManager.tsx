import React, { useState } from 'react';
import { useForm, Controller } from 'react-hook-form';
import { motion } from 'framer-motion';
import {
  CurrencyDollarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  BellIcon,
  ShieldCheckIcon,
  PencilSquareIcon,
  TrashIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';
import { BudgetInfo } from '@/types';

interface BudgetFormData {
  monthlyLimit: number;
  dailyLimit?: number;
  alertThreshold: number;
  enableAlerts: boolean;
  enableHardStop: boolean;
  emergencyContact?: string;
}

interface BudgetManagerProps {
  budgetInfo: BudgetInfo;
  onUpdateBudget: (data: BudgetFormData) => Promise<void>;
  className?: string;
}

interface BudgetAlert {
  id: string;
  type: 'warning' | 'danger' | 'info';
  message: string;
  actionRequired: boolean;
  timestamp: string;
}

export const BudgetManager: React.FC<BudgetManagerProps> = ({
  budgetInfo,
  onUpdateBudget,
  className = '',
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [saving, setSaving] = useState(false);

  // Mock alerts data
  const mockAlerts: BudgetAlert[] = [
    {
      id: '1',
      type: 'warning',
      message: 'Daily spend exceeded average by 25%',
      actionRequired: false,
      timestamp: '2025-01-12T10:30:00Z',
    },
    {
      id: '2',
      type: 'info',
      message: 'Monthly budget tracking on schedule',
      actionRequired: false,
      timestamp: '2025-01-12T09:15:00Z',
    },
  ];

  const {
    control,
    handleSubmit,
    formState: { errors, isDirty },
    reset,
    watch,
  } = useForm<BudgetFormData>({
    defaultValues: {
      monthlyLimit: budgetInfo.monthly_limit,
      dailyLimit: budgetInfo.monthly_limit / 30, // Estimate daily from monthly
      alertThreshold: 80, // 80% threshold
      enableAlerts: true,
      enableHardStop: false,
      emergencyContact: '',
    },
  });

  const watchedValues = watch();

  const onSubmit = async (data: BudgetFormData) => {
    setSaving(true);
    try {
      await onUpdateBudget(data);
      setIsEditing(false);
      reset(data);
    } catch (error) {
      console.error('Failed to update budget:', error);
    } finally {
      setSaving(false);
    }
  };

  const handleCancel = () => {
    reset();
    setIsEditing(false);
  };

  const getBudgetStatusColor = () => {
    if (budgetInfo.budget_utilization_percentage >= 90) return 'text-red-600 dark:text-red-400';
    if (budgetInfo.budget_utilization_percentage >= 75) return 'text-yellow-600 dark:text-yellow-400';
    return 'text-green-600 dark:text-green-400';
  };

  const getBudgetBarColor = () => {
    if (budgetInfo.budget_utilization_percentage >= 90) return 'bg-red-500';
    if (budgetInfo.budget_utilization_percentage >= 75) return 'bg-yellow-500';
    return 'bg-green-500';
  };

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Budget Overview Card */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
      >
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Budget Overview
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
              Current month spending and limits
            </p>
          </div>
          
          {!isEditing && (
            <button
              onClick={() => setIsEditing(true)}
              className="flex items-center gap-2 px-3 py-2 text-sm bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <PencilSquareIcon className="h-4 w-4" />
              Edit Budget
            </button>
          )}
        </div>

        {/* Budget Progress */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Monthly Budget Progress
            </span>
            <span className={`text-sm font-semibold ${getBudgetStatusColor()}`}>
              ${budgetInfo.current_month_spend.toFixed(2)} / ${budgetInfo.monthly_limit.toFixed(2)}
            </span>
          </div>
          
          <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3">
            <div
              className={`h-3 rounded-full transition-all duration-500 ${getBudgetBarColor()}`}
              style={{ width: `${Math.min(budgetInfo.budget_utilization_percentage, 100)}%` }}
            />
          </div>
          
          <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
            <span>0%</span>
            <span>{budgetInfo.budget_utilization_percentage.toFixed(1)}% used</span>
            <span>100%</span>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              ${budgetInfo.remaining_budget.toFixed(2)}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">Remaining</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-gray-900 dark:text-white">
              ${budgetInfo.projected_month_end_spend.toFixed(2)}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">Projected Total</p>
          </div>
          <div className="text-center">
            <p className={`text-2xl font-bold ${
              budgetInfo.projected_month_end_spend > budgetInfo.monthly_limit
                ? 'text-red-600 dark:text-red-400'
                : 'text-green-600 dark:text-green-400'
            }`}>
              {budgetInfo.projected_month_end_spend > budgetInfo.monthly_limit ? '+' : ''}
              ${(budgetInfo.projected_month_end_spend - budgetInfo.monthly_limit).toFixed(2)}
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {budgetInfo.projected_month_end_spend > budgetInfo.monthly_limit ? 'Over' : 'Under'} Budget
            </p>
          </div>
        </div>

        {/* Budget Alerts */}
        {(budgetInfo.alerts.approaching_limit || budgetInfo.alerts.over_budget || budgetInfo.alerts.high_daily_spend) && (
          <div className="mt-6 pt-6 border-t border-gray-200 dark:border-gray-700">
            <div className="flex items-center gap-2 mb-3">
              <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" />
              <span className="font-medium text-gray-900 dark:text-white">Active Alerts</span>
            </div>
            
            <div className="space-y-2">
              {budgetInfo.alerts.approaching_limit && (
                <div className="flex items-center gap-2 p-3 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg">
                  <BellIcon className="h-4 w-4 text-yellow-600 dark:text-yellow-400" />
                  <span className="text-sm text-yellow-700 dark:text-yellow-300">
                    Approaching monthly budget limit ({budgetInfo.budget_utilization_percentage.toFixed(1)}% used)
                  </span>
                </div>
              )}
              
              {budgetInfo.alerts.over_budget && (
                <div className="flex items-center gap-2 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <ExclamationTriangleIcon className="h-4 w-4 text-red-600 dark:text-red-400" />
                  <span className="text-sm text-red-700 dark:text-red-300">
                    Monthly budget exceeded
                  </span>
                </div>
              )}
              
              {budgetInfo.alerts.high_daily_spend && (
                <div className="flex items-center gap-2 p-3 bg-orange-50 dark:bg-orange-900/20 rounded-lg">
                  <BellIcon className="h-4 w-4 text-orange-600 dark:text-orange-400" />
                  <span className="text-sm text-orange-700 dark:text-orange-300">
                    Daily spending above average
                  </span>
                </div>
              )}
            </div>
          </div>
        )}
      </motion.div>

      {/* Budget Configuration Form */}
      {isEditing && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
        >
          <div className="flex items-center gap-2 mb-6">
            <Cog6ToothIcon className="h-5 w-5 text-gray-600 dark:text-gray-400" />
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Budget Configuration
            </h3>
          </div>

          <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Monthly Limit */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Monthly Budget Limit
                </label>
                <Controller
                  name="monthlyLimit"
                  control={control}
                  rules={{ 
                    required: 'Monthly limit is required',
                    min: { value: 1, message: 'Must be at least $1' },
                    max: { value: 10000, message: 'Must be less than $10,000' },
                  }}
                  render={({ field }) => (
                    <div className="relative">
                      <CurrencyDollarIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                      <input
                        {...field}
                        type="number"
                        step="0.01"
                        min="1"
                        max="10000"
                        className="w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                        placeholder="100.00"
                      />
                    </div>
                  )}
                />
                {errors.monthlyLimit && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                    {errors.monthlyLimit.message}
                  </p>
                )}
              </div>

              {/* Daily Limit */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Daily Budget Limit (Optional)
                </label>
                <Controller
                  name="dailyLimit"
                  control={control}
                  rules={{ 
                    min: { value: 0.01, message: 'Must be at least $0.01' },
                    validate: (value) => {
                      if (value && value * 30 > watchedValues.monthlyLimit) {
                        return 'Daily limit would exceed monthly limit';
                      }
                    }
                  }}
                  render={({ field }) => (
                    <div className="relative">
                      <CurrencyDollarIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
                      <input
                        {...field}
                        type="number"
                        step="0.01"
                        min="0.01"
                        className="w-full pl-10 pr-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                        placeholder="10.00"
                      />
                    </div>
                  )}
                />
                {errors.dailyLimit && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                    {errors.dailyLimit.message}
                  </p>
                )}
              </div>

              {/* Alert Threshold */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Alert Threshold (%)
                </label>
                <Controller
                  name="alertThreshold"
                  control={control}
                  rules={{ 
                    required: 'Alert threshold is required',
                    min: { value: 50, message: 'Must be at least 50%' },
                    max: { value: 95, message: 'Must be less than 95%' },
                  }}
                  render={({ field }) => (
                    <div className="relative">
                      <input
                        {...field}
                        type="number"
                        min="50"
                        max="95"
                        className="w-full pr-8 pl-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                        placeholder="80"
                      />
                      <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400">%</span>
                    </div>
                  )}
                />
                {errors.alertThreshold && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                    {errors.alertThreshold.message}
                  </p>
                )}
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Get notified when {watchedValues.alertThreshold}% of budget is used
                </p>
              </div>

              {/* Emergency Contact */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Emergency Contact (Optional)
                </label>
                <Controller
                  name="emergencyContact"
                  control={control}
                  rules={{ 
                    pattern: {
                      value: /^[^\s@]+@[^\s@]+\.[^\s@]+$/,
                      message: 'Please enter a valid email address'
                    }
                  }}
                  render={({ field }) => (
                    <input
                      {...field}
                      type="email"
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent dark:bg-gray-700 dark:text-white"
                      placeholder="admin@company.com"
                    />
                  )}
                />
                {errors.emergencyContact && (
                  <p className="mt-1 text-sm text-red-600 dark:text-red-400">
                    {errors.emergencyContact.message}
                  </p>
                )}
              </div>
            </div>

            {/* Checkboxes */}
            <div className="space-y-4">
              <Controller
                name="enableAlerts"
                control={control}
                render={({ field: { value, onChange } }) => (
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={value}
                      onChange={onChange}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 dark:border-gray-600 rounded"
                    />
                    <span className="ml-3 text-sm font-medium text-gray-700 dark:text-gray-300">
                      Enable budget alerts
                    </span>
                  </label>
                )}
              />

              <Controller
                name="enableHardStop"
                control={control}
                render={({ field: { value, onChange } }) => (
                  <label className="flex items-center">
                    <input
                      type="checkbox"
                      checked={value}
                      onChange={onChange}
                      className="h-4 w-4 text-red-600 focus:ring-red-500 border-gray-300 dark:border-gray-600 rounded"
                    />
                    <span className="ml-3 text-sm font-medium text-gray-700 dark:text-gray-300">
                      Enable hard stop (block requests when budget exceeded)
                    </span>
                  </label>
                )}
              />
              
              {watchedValues.enableHardStop && (
                <div className="ml-7 p-3 bg-red-50 dark:bg-red-900/20 rounded-lg">
                  <div className="flex items-center gap-2">
                    <ExclamationTriangleIcon className="h-4 w-4 text-red-600 dark:text-red-400" />
                    <span className="text-sm text-red-700 dark:text-red-300 font-medium">
                      Warning: Hard stop will completely block AI requests when budget is exceeded
                    </span>
                  </div>
                </div>
              )}
            </div>

            {/* Action Buttons */}
            <div className="flex items-center justify-end gap-3 pt-6 border-t border-gray-200 dark:border-gray-700">
              <button
                type="button"
                onClick={handleCancel}
                className="px-4 py-2 text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 rounded-lg hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
              >
                Cancel
              </button>
              <button
                type="submit"
                disabled={!isDirty || saving}
                className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {saving ? (
                  <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent" />
                ) : (
                  <CheckCircleIcon className="h-4 w-4" />
                )}
                {saving ? 'Saving...' : 'Save Budget Settings'}
              </button>
            </div>
          </form>
        </motion.div>
      )}

      {/* Recent Alerts */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Recent Alerts
          </h3>
          <button className="text-sm text-blue-600 dark:text-blue-400 hover:text-blue-700 dark:hover:text-blue-300">
            View All
          </button>
        </div>

        <div className="space-y-3">
          {mockAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`flex items-start gap-3 p-3 rounded-lg ${
                alert.type === 'warning'
                  ? 'bg-yellow-50 dark:bg-yellow-900/20'
                  : alert.type === 'danger'
                  ? 'bg-red-50 dark:bg-red-900/20'
                  : 'bg-blue-50 dark:bg-blue-900/20'
              }`}
            >
              {alert.type === 'warning' ? (
                <ExclamationTriangleIcon className="h-5 w-5 text-yellow-600 dark:text-yellow-400 mt-0.5" />
              ) : alert.type === 'danger' ? (
                <ExclamationTriangleIcon className="h-5 w-5 text-red-600 dark:text-red-400 mt-0.5" />
              ) : (
                <CheckCircleIcon className="h-5 w-5 text-blue-600 dark:text-blue-400 mt-0.5" />
              )}
              
              <div className="flex-1">
                <p className={`text-sm font-medium ${
                  alert.type === 'warning'
                    ? 'text-yellow-700 dark:text-yellow-300'
                    : alert.type === 'danger'
                    ? 'text-red-700 dark:text-red-300'
                    : 'text-blue-700 dark:text-blue-300'
                }`}>
                  {alert.message}
                </p>
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  {new Date(alert.timestamp).toLocaleString()}
                </p>
              </div>
              
              {alert.actionRequired && (
                <button className="text-xs bg-gray-200 dark:bg-gray-600 text-gray-700 dark:text-gray-300 px-2 py-1 rounded">
                  Action Required
                </button>
              )}
            </div>
          ))}
        </div>
      </motion.div>
    </div>
  );
};

export default BudgetManager;