import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  scrollBehavior: () => ({ top: 0 }),
  routes: [
    { path: '/', redirect: '/workspace' },
    {
      path: '/workspace',
      name: 'workspace',
      component: () => import('@/views/WorkspaceView.vue'),
      meta: { title: '工作台' },
    },
    {
      path: '/review/new',
      name: 'review-new',
      component: () => import('@/views/ReviewWizardView.vue'),
      meta: { title: '新建审查' },
    },
    {
      path: '/review/:sid',
      name: 'review',
      component: () => import('@/views/ReviewResultView.vue'),
      meta: { title: '审查结果' },
    },
    {
      path: '/scenarios',
      name: 'scenarios',
      component: () => import('@/views/ScenariosView.vue'),
      meta: { title: '场景库' },
    },
    {
      path: '/scenarios/:sid',
      name: 'scenario',
      component: () => import('@/views/ScenarioDetailView.vue'),
      meta: { title: '场景详情' },
    },
    {
      path: '/benchmark',
      name: 'benchmark',
      component: () => import('@/views/BenchmarkView.vue'),
      meta: { title: '评测' },
    },
    {
      path: '/llm',
      name: 'llm',
      component: () => import('@/views/LlmView.vue'),
      meta: { title: 'LLM 治理' },
    },
    {
      path: '/system',
      name: 'system',
      component: () => import('@/views/SystemView.vue'),
      meta: { title: '系统' },
    },
    { path: '/:pathMatch(.*)*', redirect: '/workspace' },
  ],
})

router.afterEach((to) => {
  const title = to.meta.title as string | undefined
  document.title = title ? `${title} · ClauseScope` : 'ClauseScope · 合同合规推理台'
})

export default router
